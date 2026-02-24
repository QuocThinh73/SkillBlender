from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import math

import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, set_seed
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

H, W = 480, 640

# Dictionary ánh xạ tên task bạn truyền vào arg sang task_id của môi trường
TASK_NAME_TO_ID = {
    'task_ball': 0,
    'task_box': 1,
    'task_button': 2,
    'task_cabinet': 3,
    'task_carry': 4,
    'task_lift': 5,
    'task_reach': 6,
    'task_transfer': 7
}

def visualize_task(eval_task_id, env):
    """Only be used when with display"""
    env.gym.clear_lines(env.viewer)
    
    if eval_task_id == 6: # REACH
        axes_geom = gymutil.AxesGeometry(0.15)
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        yellow_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 1, 0))
        purple_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 0, 1))
        
        wrist_pos = env.rigid_state[:, env.wrist_indices, :7] 
        ref_wrist_pos = env.ref_wrist_pos 
        for i in range(env.num_envs):
            for j in range(2):
                wrist_pos_ij = gymapi.Transform(gymapi.Vec3(wrist_pos[i, j, 0], wrist_pos[i, j, 1], wrist_pos[i, j, 2]), gymapi.Quat())
                ref_wrist_pos_ij = gymapi.Transform(gymapi.Vec3(ref_wrist_pos[i, j, 0], ref_wrist_pos[i, j, 1], ref_wrist_pos[i, j, 2]), gymapi.Quat())
                ori_wrist_pos_ij = gymapi.Transform(gymapi.Vec3(env.ori_wrist_pos[i, j, 0], env.ori_wrist_pos[i, j, 1], env.ori_wrist_pos[i, j, 2]), gymapi.Quat())
                
                gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[i], wrist_pos_ij)
                gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], ref_wrist_pos_ij)
                gymutil.draw_lines(purple_geom, env.gym, env.viewer, env.envs[i], ori_wrist_pos_ij)
                
    elif eval_task_id in [1, 7]: # BOX + TRANSFER
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        red_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            box_goal_pos = env.small_box_goal_pos[i, :3]
            box_goal_i = gymapi.Transform(gymapi.Vec3(box_goal_pos[0], box_goal_pos[1], box_goal_pos[2]), gymapi.Quat())
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], box_goal_i)
            
    elif eval_task_id == 2: # BUTTON
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        red_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            button_goal_pos = env.button_goal_pos[i, :3]
            button_goal_i = gymapi.Transform(gymapi.Vec3(button_goal_pos[0], button_goal_pos[1], button_goal_pos[2]), gymapi.Quat())
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], button_goal_i)
            
    elif eval_task_id in [4, 5]: # LIFT + CARRY
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        yellow_geom = gymutil.WireframeSphereGeometry(0.1, 12, 12, sphere_pose, color=(1, 1, 0))
        red_geom = gymutil.WireframeSphereGeometry(0.1, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            box_pos = env.big_box_root_states[i, :3]
            box_goal_pos = env.big_box_goal_pos[i]
            gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(box_pos[0], box_pos[1], box_pos[2]), gymapi.Quat()))
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(box_goal_pos[0], box_goal_pos[1], box_goal_pos[2]), gymapi.Quat()))
            
    elif eval_task_id == 0: # BALL
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        red_geom = gymutil.WireframeSphereGeometry(0.2, 12, 12, sphere_pose, color=(1, 0, 0))
        for i in range(env.num_envs):
            goal_pos = env.ball_goal_pos[i, :3] 
            gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2]), gymapi.Quat()))
            
    elif eval_task_id == 3: # CABINET
        pass 

def get_camera_pose(eval_task_id):
    if not EGO_CENTRIC:
        # ID 2: Button, ID 0: Ball, ID 3: Cabinet
        if eval_task_id in [0, 2, 3]:
            camera_offset = gymapi.Vec3(-1, -2, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(45))
        else:
            camera_offset = gymapi.Vec3(1, -1, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
    else:
        camera_offset = gymapi.Vec3(0.1, 0, 0.9)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(45))
    return gymapi.Transform(camera_offset, camera_rotation)
    
def override_env_cfg(env_cfg: LeggedRobotCfg, args, eval_task_id: int):
    print('====> URDF file:', env_cfg.asset.file)
    
    # 1. Ép môi trường chỉ tạo đúng 1 env cho task đang muốn test
    env_cfg.env.num_task_ball_envs = 1 if eval_task_id == 0 else 0
    env_cfg.env.num_task_box_envs = 1 if eval_task_id == 1 else 0
    env_cfg.env.num_task_button_envs = 1 if eval_task_id == 2 else 0
    env_cfg.env.num_task_cabinet_envs = 1 if eval_task_id == 3 else 0
    env_cfg.env.num_task_carry_envs = 1 if eval_task_id == 4 else 0
    env_cfg.env.num_task_lift_envs = 1 if eval_task_id == 5 else 0
    env_cfg.env.num_task_reach_envs = 1 if eval_task_id == 6 else 0
    env_cfg.env.num_task_transfer_envs = 1 if eval_task_id == 7 else 0
    
    env_cfg.env.num_envs = 1 # Tổng số env luôn là 1 để visualize

    # 2. Cài đặt episode length tương ứng
    if eval_task_id == 6: # REACH
        env_cfg.env.episode_length_s = 20
        env_cfg.human.freq = 2
    elif eval_task_id in [1, 2, 7]: # BOX, BUTTON, TRANSFER
        env_cfg.env.episode_length_s = 2.5
    elif eval_task_id in [0, 3, 4, 5]: # BALL, CABINET, CARRY, LIFT
        env_cfg.env.episode_length_s = 2.5
    else:
        env_cfg.env.episode_length_s = 8

    # 3. Tắt nhiễu và curriculum để test cho mượt
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    return env_cfg

def play(args):
    # Lấy eval_task từ arg (Ví dụ: truyền --task h1_unified_task --eval_task task_button)
    # Vì file train dùng chung `args.task` là "h1_unified_task", ta sẽ thêm một logic để tách
    eval_task_name = getattr(args, 'eval_task', 'task_ball') # Mặc định là ball nếu không truyền
    if eval_task_name not in TASK_NAME_TO_ID:
        raise ValueError(f"Invalid eval_task: {eval_task_name}. Choose from {list(TASK_NAME_TO_ID.keys())}")
    
    eval_task_id = TASK_NAME_TO_ID[eval_task_name]
    print(f"==================================================")
    print(f"VISUALIZING TASK: {eval_task_name.upper()} (ID: {eval_task_id})")
    print(f"==================================================")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run, experiment_name=args.experiment_name)
    
    # Ghi đè cấu hình cho đúng 1 task
    env_cfg = override_env_cfg(env_cfg, args, eval_task_id)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Lấy task_ids từ môi trường (Lúc này tensor này sẽ có shape [1] và giá trị bằng eval_task_id)
    task_ids = env.task_ids

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.run_name = 'play'

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # Lấy Multi-Task Policy
    policy = ppo_runner.get_inference_policy(device=env.device, hrl=False)
    
    model_name = f'{args.load_run}_{train_cfg.runner.resume_path.split("_")[-1].split(".")[0]}'

    robot_index = 0 # Chỉ có 1 robot
    rew_log_interval = env.max_episode_length - 1 
    N_rollouts = 10
    
    if RECORD_FRAMES:
        frame_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
        os.makedirs(frame_path, exist_ok=True)
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = W
        camera_properties.height = H
        cam = env.gym.create_camera_sensor(env.envs[robot_index], camera_properties)
        
        camera_pose = get_camera_pose(eval_task_id)
        
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
    if eval_task_id == 6: # REACH
        last_ref_wrist_pos = env.ref_wrist_pos[robot_index][:,:3].cpu().numpy()
    if eval_task_id != 6: # Các task high-level khác
        max_steps = int(env.max_episode_length) * 10

    for i_rollout in range(N_rollouts):
        print(f"====> Rollout {i_rollout+1}/{N_rollouts}")
        logger = Logger(env.dt)
        env.ori_root_states = env.root_states.clone()
        if RECORD_FRAMES:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            filename_mp4 = f'{eval_task_name}_{model_name}_{i_rollout}.mp4'
            video = cv2.VideoWriter(os.path.join(frame_path, filename_mp4), fourcc, 25.0, (W, H))
            
        for i in tqdm(range(max_steps)):
            visualize_task(eval_task_id, env)

            # --- TRUYỀN task_ids VÀO POLICY ---
            actions = policy(obs.detach(), task_ids=task_ids)

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
            
            # ... (Phần log states và metrics của bạn giữ nguyên, 
            # tuy nhiên hãy sửa các if check args.task thành kiểm tra eval_task_id) ...

        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    EGO_CENTRIC = False
    FIX_COMMAND = True
    
    # 1. Trick để trích xuất arg --eval_task thủ công
    eval_task = 'task_ball' # Giá trị mặc định
    if '--eval_task' in sys.argv:
        idx = sys.argv.index('--eval_task')
        eval_task = sys.argv[idx + 1]
        
        # XOÁ ARGUMENT NÀY ĐỂ get_args() KHÔNG BỊ CRASH
        sys.argv.pop(idx) # Xóa value ('task_button')
        sys.argv.pop(idx) # Xóa key ('--eval_task')
        
    # 2. Gọi get_args gốc (lúc này nó sẽ không thấy --eval_task nữa)
    args = get_args(test=True)
    
    # 3. Gắn biến đã lưu vào args để truyền xuống cho hàm play()
    args.eval_task = eval_task
    
    play(args)