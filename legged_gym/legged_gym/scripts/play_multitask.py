import math
import os
import cv2
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from tqdm import tqdm


H, W = 480, 640

def visualize_task(env):
    env.gym.clear_lines(env.viewer)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    red_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))    
    for i in range(env.num_envs):
        # Task reach
        left_wrist_goal_pos = env.wrist_goal_pos[i, 0, :3]
        right_wrist_goal_pos = env.wrist_goal_pos[i, 1, :3]
        left_wrist_goal_i = gymapi.Transform(gymapi.Vec3(left_wrist_goal_pos[0], left_wrist_goal_pos[1], left_wrist_goal_pos[2]), gymapi.Quat())
        right_wrist_goal_i = gymapi.Transform(gymapi.Vec3(right_wrist_goal_pos[0], right_wrist_goal_pos[1], right_wrist_goal_pos[2]), gymapi.Quat())
        gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], left_wrist_goal_i)
        gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], right_wrist_goal_i)
        # Task button
        button_goal_pos = env.button_goal_pos[i, :3]
        button_goal_i = gymapi.Transform(gymapi.Vec3(button_goal_pos[0], button_goal_pos[1], button_goal_pos[2]), gymapi.Quat())
        gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], button_goal_i)
        # Task cabinet
        # Task box
        small_box_goal_pos = env.small_box_goal_pos[i, :3]
        small_box_goal_i = gymapi.Transform(gymapi.Vec3(small_box_goal_pos[0], small_box_goal_pos[1], small_box_goal_pos[2]), gymapi.Quat())
        gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], small_box_goal_i)
        # Task ball
        ball_goal_pos = env.ball_goal_pos[i, :3]
        ball_goal_i = gymapi.Transform(gymapi.Vec3(ball_goal_pos[0], ball_goal_pos[1], ball_goal_pos[2]), gymapi.Quat())
        gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], ball_goal_i)

def override_env_cfg(env_cfg, args):
    default_num_envs = 50
    if args.visualize:
        default_num_envs = 1
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, default_num_envs)

    # env_cfg.env.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    # env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    return env_cfg

def get_camera_pose():
    if not EGO_CENTRIC:
        camera_offset = gymapi.Vec3(-1, -2, 1)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                        np.deg2rad(45))
    else:
        camera_offset = gymapi.Vec3(0.1, 0, 0.9)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0),
                                                    np.deg2rad(45))
    return gymapi.Transform(camera_offset, camera_rotation)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run, experiment_name=args.experiment_name)
    env_cfg = override_env_cfg(env_cfg, args)
    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    train_cfg.runner.run_name = "play"

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device, hrl=True)

    model_name = f'{args.load_run}_{train_cfg.runner.resume_path.split("_")[-1].split(".")[0]}'

    robot_index = 0
    N_rollouts = 10

    if RECORD_FRAMES:
        frame_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
        os.makedirs(frame_path, exist_ok=True)
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = W
        camera_properties.height = H
        cam = env.gym.create_camera_sensor(env.envs[robot_index], camera_properties)
        camera_pose = get_camera_pose()
        actor_handle = env.gym.get_actor_handle(env.envs[robot_index], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[robot_index], actor_handle, 0)
        env.gym.attach_camera_to_body(
            cam, 
            env.envs[robot_index], 
            body_handle,
            camera_pose,
            gymapi.FOLLOW_POSITION if not EGO_CENTRIC else gymapi.FOLLOW_TRANSFORM
        )

    max_steps = int(env.max_episode_length)

    for i_rollout in range(N_rollouts):
        if RECORD_FRAMES:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            filename_mp4 = f'{args.task}_{model_name}_{i_rollout}.mp4'
            video = cv2.VideoWriter(os.path.join(frame_path, filename_mp4), fourcc, 25.0, (W, H))

        for i in tqdm(range(max_steps)):
            visualize_task(env)

            actions = policy(obs.detach())
            actions = actions["actions_mean"]

            if FIX_COMMAND:
                env.commands[:, 0] = 1.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
                env.commands[:, 3] = 0.0

            obs, _, rews, dones, infos = env.step(actions.detach())


            if RECORD_FRAMES:
                if i % 4 == 0:
                    env.gym.fetch_results(env.sim, True)
                    env.gym.step_graphics(env.sim)
                    env.gym.render_all_camera_sensors(env.sim)
                    img = env.gym.get_camera_image(env.sim, env.envs[robot_index], cam, gymapi.IMAGE_COLOR)
                    img = np.reshape(img, (H, W, 4))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img[..., :3])
            
            video.release()

if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    EGO_CENTRIC = False
    FIX_COMMAND = True
    args = get_args(test=True)
    play(args)