# SPDX-License-Identifier: BSD-3-Clause
import os
import math
from typing import Optional

import isaacgym  # noqa: F401 (MUST be before torch)
import torch

from isaacgym import gymapi, gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry

_SUBTASK_CHOICES = ["ball", "box", "button", "cabinet", "carry", "lift", "reach", "transfer"]


def get_args_play_unified_fixed():
    from isaacgym import gymutil

    custom_parameters = [
        {"name": "--task", "type": str, "default": "h1", "help": "Task name (env name)."},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--resume_stop_at_max", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str, "help": "Experiment name."},
        {"name": "--run_name", "type": str, "required": False},
        {"name": "--entity", "type": str, "required": False},
        {"name": "--load_run", "type": str, "default": "", "help": "Run folder to load."},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Checkpoint. -1 = last."},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--stochastic", "action": "store_true", "default": False},
        {"name": "--use_jit", "action": "store_true", "default": False},
        {"name": "--wandb", "type": str, "default": ""},
        {"name": "--visualize", "action": "store_true", "default": False},
        {"name": "--baseline", "type": str, "default": "None"},
        # ✅ subtask BẮT BUỘC
        {
            "name": "--subtask",
            "type": str,
            "default": "",
            "help": f"REQUIRED. One of: {', '.join(_SUBTASK_CHOICES)}",
        },
    ]

    args = gymutil.parse_arguments(description="RL Policy (unified play FIXED subtask)", custom_parameters=custom_parameters)
    args.test = True

    if not args.subtask or args.subtask.lower().strip() not in _SUBTASK_CHOICES:
        raise ValueError(f"--subtask is REQUIRED. Choose one of: {_SUBTASK_CHOICES}")

    # same alignment as legged_gym.utils.get_args
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def _subtask_to_id(env, subtask: str) -> int:
    st = subtask.lower().strip()
    t = env.cfg.task
    mapping = {
        "ball": t.TASK_BALL,
        "box": t.TASK_BOX,
        "button": t.TASK_BUTTON,
        "cabinet": t.TASK_CABINET,
        "carry": t.TASK_CARRY,
        "lift": t.TASK_LIFT,
        "reach": t.TASK_REACH,
        "transfer": t.TASK_TRANSFER,
    }
    return int(mapping[st])


def override_env_cfg_for_play(env_cfg, args):
    # giống style file gốc: giảm env để nhìn, tắt noise/domain rand
    if getattr(args, "visualize", False):
        env_cfg.env.num_envs = 1
    else:
        # play thường không cần 4096 env
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)

    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_friction = False

    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5

    if args.subtask == "reach":
        env_cfg.env.episode_length_s = 20
        env_cfg.human.freq = 2
    elif args.subtask in ["box", "transfer", "button", "lift", "ball", "carry"]:
        env_cfg.env.episode_length_s = 2.5
    elif args.subtask in ["cabinet"]:
        env_cfg.env.episode_length_s = 2.5

    return env_cfg


def visualize_unified(subtask: str, env):
    """Vẽ debug overlay (viewer) theo đúng tensor/field của H1UnifiedTask."""
    if env.viewer is None:
        return

    env.gym.clear_lines(env.viewer)

    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)

    # goal markers
    red = gymutil.WireframeSphereGeometry(0.10, 12, 12, sphere_pose, color=(1, 0, 0))
    yellow = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 1, 0))
    purple = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 1))
    axes = gymutil.AxesGeometry(0.15)

    st = subtask.lower().strip()

    if st == "ball":
        for i in range(env.num_envs):
            gp = env.ball_goal_pos[i, :3]
            T = gymapi.Transform(gymapi.Vec3(gp[0], gp[1], gp[2]), gymapi.Quat())
            gymutil.draw_lines(red, env.gym, env.viewer, env.envs[i], T)

    elif st in ("box", "transfer"):
        for i in range(env.num_envs):
            gp = env.small_box_goal_pos[i, :3]
            T = gymapi.Transform(gymapi.Vec3(gp[0], gp[1], gp[2]), gymapi.Quat())
            gymutil.draw_lines(red, env.gym, env.viewer, env.envs[i], T)

    elif st == "button":
        for i in range(env.num_envs):
            gp = env.button_goal_pos[i, :3]
            T = gymapi.Transform(gymapi.Vec3(gp[0], gp[1], gp[2]), gymapi.Quat())
            gymutil.draw_lines(red, env.gym, env.viewer, env.envs[i], T)

    elif st in ("carry", "lift"):
        for i in range(env.num_envs):
            gp = env.big_box_goal_pos[i, :3]
            T = gymapi.Transform(gymapi.Vec3(gp[0], gp[1], gp[2]), gymapi.Quat())
            gymutil.draw_lines(red, env.gym, env.viewer, env.envs[i], T)

    elif st == "reach":
        # current wrist + ref wrist + ori wrist
        wrist_pos = env.rigid_state[:, env.wrist_indices, :3]          # [N,2,3]
        ref_wrist = env.ref_wrist_pos[:, :, :3]                        # [N,2,3]
        ori_wrist = env.ori_wrist_pos[:, :, :3]                        # [N,2,3]
        for i in range(env.num_envs):
            for j in range(2):
                cur = wrist_pos[i, j]
                ref = ref_wrist[i, j]
                ori = ori_wrist[i, j]
                gymutil.draw_lines(axes, env.gym, env.viewer, env.envs[i],
                                   gymapi.Transform(gymapi.Vec3(cur[0], cur[1], cur[2]), gymapi.Quat()))
                gymutil.draw_lines(yellow, env.gym, env.viewer, env.envs[i],
                                   gymapi.Transform(gymapi.Vec3(ref[0], ref[1], ref[2]), gymapi.Quat()))
                gymutil.draw_lines(purple, env.gym, env.viewer, env.envs[i],
                                   gymapi.Transform(gymapi.Vec3(ori[0], ori[1], ori[2]), gymapi.Quat()))

    elif st == "cabinet":
        # task này bạn nói “no need visualize” ở file gốc → để trống
        pass


def main():
    args = get_args_play_unified_fixed()

    # cfgs
    env_cfg, train_cfg = task_registry.get_cfgs(
        name=args.task, load_run=args.load_run, experiment_name=args.experiment_name
    )

    env_cfg = override_env_cfg_for_play(env_cfg, args)

    # env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    fixed_id = _subtask_to_id(env, args.subtask)

    # ✅ Monkeypatch reset_idx để KHÓA task_id ngay từ reset (và sample goal đúng task đó)
    def reset_idx_fixed(env_ids):
        if env_ids.numel() == 0:
            return
        env.task_ids[env_ids] = fixed_id

        # gọi base reset (sẽ gọi env._reset_root_states / _reset_dofs của bạn)
        LeggedRobot.reset_idx(env, env_ids)

        # sample goals đúng task (ball_goal_pos/small_box_goal_pos/...)
        env._sample_goals(env_ids)

        # clear stacked histories
        for i in range(env.obs_history.maxlen):
            env.obs_history[i][env_ids] *= 0
        for i in range(env.critic_history.maxlen):
            env.critic_history[i][env_ids] *= 0

    env.reset_idx = reset_idx_fixed

    # force ngay từ đầu cho toàn bộ env
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(all_ids)
    obs = env.get_observations()

    print(obs)

    print(f"[INFO] Playing FIXED subtask='{args.subtask}' (id={fixed_id}).")

    # runner/policy
    train_cfg.runner.resume = True
    train_cfg.runner.run_name = "play"

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # unified của bạn dùng ActorCriticHierarchical => inference policy có thể trả dict
    policy = ppo_runner.get_inference_policy(device=env.device, hrl=True)

    # ✅ commands giống env mới:
    # - Unified task của bạn không dùng walking (command ranges = 0 hết)
    # - nên set all command = 0 để obs ổn định
    env.commands[:] = 0.0

    max_steps = int(env.max_episode_length) * 10

    with torch.no_grad():
        for _ in range(max_steps):
            # visualize (chỉ có tác dụng khi headless=False)
            visualize_unified(args.subtask, env)

            out = policy(obs)

            # lấy action đúng (HRL thường trả dict)
            if isinstance(out, dict):
                actions = out.get("actions_mean", None)
                if actions is None:
                    actions = out.get("actions", None)
                if actions is None:
                    actions = next(v for v in out.values() if torch.is_tensor(v))
            else:
                actions = out

            obs, _, _, _, _ = env.step(actions)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
