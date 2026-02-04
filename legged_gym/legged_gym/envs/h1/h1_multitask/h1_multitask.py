from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from .h1_multitask_config import H1MultitaskCfg

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot, LEGGED_GYM_ROOT_DIR, get_euler_xyz_tensor, quat_rotate_inverse

import random
import os
import numpy as np
from collections import deque


class H1Multitask(LeggedRobot):
    def __init__(self, cfg: H1MultitaskCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg

        self.num_tasks = cfg.env.num_tasks
        self.TASK_REACH = cfg.env.TASK_REACH
        self.TASK_BUTTON = cfg.env.TASK_BUTTON
        self.TASK_CABINET = cfg.env.TASK_CABINET
        self.TASK_BOX = cfg.env.TASK_BOX
        self.TASK_BALL = cfg.env.TASK_BALL
        self.task_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Goals
        ## Task reach
        self.wrist_goal_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        ## Task button
        self.button_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        ## Task cabinet
        self.cabinet_dof_goal = 0
        ## Task box
        self.small_box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        ## Task ball
        self.ball_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task conditioning
        ## Fixed chain order
        self.task_chain = torch.tensor(
            [self.TASK_REACH, self.TASK_BUTTON, self.TASK_CABINET, self.TASK_BOX, self.TASK_BALL],
            device=self.device, dtype=torch.long
        )
        self.num_chain = self.task_chain.numel()
        ## Convert length to max steps
        self.chain_max_task_length = torch.tensor([
            np.ceil(self.cfg.env.reach_length_s / self.dt),
            np.ceil(self.cfg.env.button_length_s / self.dt),
            np.ceil(self.cfg.env.cabinet_length_s / self.dt),
            np.ceil(self.cfg.env.box_length_s / self.dt),
            np.ceil(self.cfg.env.ball_length_s / self.dt),
        ], device=self.device, dtype=torch.long)
        ## Buffers for switch task
        self.task_ptr = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) 
        self.switch_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.task_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_task_length = self.chain_max_task_length[self.task_ptr].clone()

        # Precompute
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.gym.simulate(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.compute_observations()

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.humanoid_root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.humanoid_root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.terrain.mesh_type == "plane":
            self._create_ground_plane()
        else:
            raise ValueError("Only support plane")
        self._create_envs()

    def _create_envs(self):
        # Load the robot URDF/MJCF asset
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(humanoid_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(humanoid_asset)

        # Save names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.envs = []
        self.actor_handles = []
        self.humanoid_idxs = []
        humanoid_pose = gymapi.Transform()
        humanoid_base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.humanoid_base_init_state = to_torch(humanoid_base_init_state_list, device=self.device, requires_grad=False)

        # Assets
        ## Task ball
        ### Ball wall assets
        goal_assets = []
        goal_pose = gymapi.Transform()
        for goal_dim in self.cfg.asset.goal_dims:
            goal_dims = gymapi.Vec3(*goal_dim)

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True

            goal_asset = self.gym.create_box(
                self.sim,
                goal_dims.x, goal_dims.y, goal_dims.z,
                asset_options
            )
            goal_assets.append(goal_asset)
        ### Ball assets
        ball_size = self.cfg.asset.ball_size
        asset_options = gymapi.AssetOptions()
        ball_asset = self.gym.create_sphere(self.sim, ball_size, asset_options)
        ball_pose = gymapi.Transform()
        self.ball_idxs = []
        ## Task button
        ### Wall assets
        wall_dims = gymapi.Vec3(*self.cfg.asset.wall_dims)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        wall_asset = self.gym.create_box(self.sim, wall_dims.x, wall_dims.y, wall_dims.z, asset_options)
        wall_pose = gymapi.Transform()
        ## Task box
        ### Table assets
        table_dims = gymapi.Vec3(*self.cfg.asset.table_dims)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        ### Small box assets
        small_box_size = self.cfg.asset.small_box_size
        asset_options = gymapi.AssetOptions()
        small_box_asset = self.gym.create_box(self.sim, small_box_size, small_box_size, small_box_size, asset_options)
        small_box_pose = gymapi.Transform()
        self.small_box_idxs = []
        ## Task cabinet
        ### Cabinet assets
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        cabinet_asset = self.gym.load_asset(self.sim, self.cfg.asset.gapartnet_root, f"{self.cfg.asset.gapartnet_id}/mobility_annotation_gapartnet.urdf", asset_options)
        self.cabinet_num_dofs = self.gym.get_asset_dof_count(cabinet_asset)
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        cabinet_default_dof_pos = np.zeros(self.cabinet_num_dofs, dtype=np.float32)
        cabinet_default_dof_pos[:] = self.cfg.asset.cabinet_dof_default
        cabinet_default_dof_state = np.zeros(self.cabinet_num_dofs, gymapi.DofState.dtype)
        cabinet_default_dof_state["pos"] = cabinet_default_dof_pos
        cabinet_dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE) # NO DOF_MODE_POS
        cabinet_dof_props["stiffness"].fill(0.0) # how fast the arti obj gonna move
        cabinet_dof_props["damping"].fill(5.0) # large damping to prevent oscillation
        cabinet_dof_props["friction"].fill(0.0)
        cabinet_pose = gymapi.Transform()
        self.cabinet_idxs = []

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            pos = self.env_origins[i].clone()
            humanoid_pose.p = gymapi.Vec3(pos[0].item(), pos[1].item(), pos[2].item())

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(humanoid_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(env_handle, humanoid_asset, humanoid_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions)
            self.actor_handles.append(actor_handle)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.humanoid_idxs.append(self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))

            # Assets
            ## Task ball
            ### Goal assets
            for goal_i, goal_asset in enumerate(goal_assets):
                offsets = self.cfg.asset.goal_offsets[goal_i]
                goal_pose.p = gymapi.Vec3(pos[0].item() + offsets[0], pos[1].item() + offsets[1], pos[2].item() + offsets[2])
                self.gym.create_actor(env_handle, goal_asset, goal_pose, f"goal_{goal_i}", i, 0)
            ### Ball assets
            ball_pose.p.x = pos[0].item()
            ball_pose.p.y = pos[1].item()
            ball_pose.p.z = pos[2].item() + 0.5 * ball_size
            ball_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-np.pi, np.pi))
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 0)
            ball_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            for prop in ball_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.ball_range_mass)
            self.ball_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))
            ## Task button
            ### Wall assets
            wall_offsets = self.cfg.asset.wall_offsets
            wall_pose.p = gymapi.Vec3(pos[0] + wall_offsets[0], pos[1] + wall_offsets[1], pos[2] + wall_offsets[2])
            self.gym.create_actor(env_handle, wall_asset, wall_pose, "wall", i, 0)
            ## Task box
            ### Table assets
            table_offsets = self.cfg.asset.table_offsets
            table_pose.p = gymapi.Vec3(pos[0] + table_offsets[0], pos[1] + table_offsets[1], pos[2] + table_offsets[2])
            self.gym.create_actor(env_handle, table_asset, table_pose, "table", i, 0)
            ### Small box assets
            small_box_pose.p.x = table_pose.p.x
            small_box_pose.p.y = table_pose.p.y
            small_box_pose.p.z = table_pose.p.z + 0.5 * small_box_size
            small_box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-np.pi, np.pi))
            small_box_handle = self.gym.create_actor(env_handle, small_box_asset, small_box_pose, "small_box", i, 0)
            self.small_box_idxs.append(self.gym.get_actor_index(env_handle, small_box_handle, gymapi.DOMAIN_SIM))
            ## Task cabinet
            ### Cabinet assets
            cabinet_offsets = self.cfg.asset.cabinet_offsets
            cabinet_pose.p = gymapi.Vec3(pos[0] + cabinet_offsets[0], pos[1] + cabinet_offsets[1], pos[2] + cabinet_offsets[2])
            cabinet_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -np.pi / 2)
            cabinet_handle = self.gym.create_actor(env_handle, cabinet_asset, cabinet_pose, "cabinet", i, 0)
            self.gym.set_actor_dof_properties(env_handle, cabinet_handle, cabinet_dof_props)
            self.gym.set_actor_dof_states(env_handle, cabinet_handle, cabinet_default_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_scale(env_handle, cabinet_handle, self.cfg.asset.cabinet_scale)
            self.cabinet_idxs.append(self.gym.get_actor_index(env_handle, cabinet_handle, gymapi.DOMAIN_SIM))
        
        self.humanoid_idxs = torch.tensor(self.humanoid_idxs, device=self.device, dtype=torch.long)
        self.ball_idxs = torch.tensor(self.ball_idxs, device=self.device, dtype=torch.long)
        self.small_box_idxs = torch.tensor(self.small_box_idxs, device=self.device, dtype=torch.long)
        self.cabinet_idxs = torch.tensor(self.cabinet_idxs, device=self.device, dtype=torch.long)
            
        # Common body parts
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        ### Other body parts
        #### Elbow
        elbow_names = [s for s in self.body_names if self.cfg.asset.elbow_name in s]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], elbow_names[i])
        #### Torso 
        torso_names = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(torso_names)):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], torso_names[i])
        #### Wrist
        wrist_names = [s for s in self.body_names if self.cfg.asset.wrist_name in s]
        self.wrist_indices = torch.zeros(len(wrist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_names)):
            self.wrist_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], wrist_names[i])

    def _init_buffers(self):
        self._init_visual_buffers()

        # Get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.humanoid_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.humanoid_idxs[0]]
        self.ball_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.ball_idxs[0]]
        self.small_box_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.small_box_idxs[0]]
        self.cabinet_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.cabinet_idxs[0]]
        self.humanoid_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:]

        self.dof_pos = self.humanoid_dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 0]
        self.dof_vel = self.humanoid_dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 1]

        self.base_quat = self.humanoid_root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.common_step_counter = 0
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.humanoid_root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.measured_heights = 0

        # Joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_privileged_obs, dtype=torch.float, device=self.device))

    def _reset_dofs(self, env_ids):
        # Reset humanoid dof states
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.0
        # Reset cabinet dof states
        self.cabinet_dof_state[env_ids, :, 0] = self.cfg.asset.cabinet_dof_default

        humanoid_ids_int32 = self.humanoid_idxs[env_ids].to(dtype=torch.int32)
        cabinet_ids_int32 = self.cabinet_idxs[env_ids].to(dtype=torch.int32)

        ids = torch.cat(
            [
                humanoid_ids_int32,
                cabinet_ids_int32
            ]
        )

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(ids), len(ids))

    def _reset_root_states(self, env_ids):
        pos = self.env_origins[env_ids].clone()
        # Reset humanoid root states
        self.humanoid_root_states[env_ids] = self.humanoid_base_init_state
        self.humanoid_root_states[env_ids, :3] += self.env_origins[env_ids]
        # Reset ball root states
        self.ball_root_states[env_ids, 0] = pos[:, 0] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.asset.ball_range_x).to(self.device)
        self.ball_root_states[env_ids, 1] = pos[:, 1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.asset.ball_range_y).to(self.device)
        self.ball_root_states[env_ids, 2] = 0.5 * self.cfg.asset.ball_size
        self.ball_root_states[env_ids, 3] = 1
        self.ball_root_states[env_ids, 4:] = 0
        # Reset ball goal
        self.ball_goal_pos[env_ids, 0] = pos[:, 0] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.ball_goal_x).to(self.device)
        self.ball_goal_pos[env_ids, 1] = pos[:, 1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.ball_goal_y).to(self.device)
        self.ball_goal_pos[env_ids, 2] = torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.ball_goal_z).to(self.device)
        # Reset small box root states
        self.small_box_root_states[env_ids, 0] = pos[:, 0]  + self.cfg.asset.table_offsets[0] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.asset.small_box_range_x).to(self.device)
        self.small_box_root_states[env_ids, 1] = pos[:, 1]  + self.cfg.asset.table_offsets[1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.asset.small_box_range_y).to(self.device)
        self.small_box_root_states[env_ids, 2] = self.cfg.asset.table_offsets[2] + 0.5 * self.cfg.asset.table_dims[2] + 0.5 * self.cfg.asset.small_box_size
        # Reset small box goal
        self.small_box_goal_pos[env_ids, 0] = self.small_box_root_states[env_ids, 0] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.small_box_x).to(self.device)
        self.small_box_goal_pos[env_ids, 1] = self.small_box_root_states[env_ids, 1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.small_box_y).to(self.device)
        self.small_box_goal_pos[env_ids, 2] = self.small_box_root_states[env_ids, 2]
        # Task button
        self.button_goal_pos[env_ids, 0] = pos[:, 0] + self.cfg.asset.wall_offsets[0]
        self.button_goal_pos[env_ids, 1] = pos[:, 1] + self.cfg.asset.wall_offsets[1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.button_goal_y).to(self.device)
        self.button_goal_pos[env_ids, 2] = self.cfg.asset.button_ori_z + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.button_goal_z).to(self.device)
        # Task reach
        center_x = pos[:, 0] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.center_goal_x).to(self.device)
        center_y = pos[:, 1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.center_goal_y).to(self.device)
        center_z = pos[:, 2] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.center_goal_z).to(self.device)
        self.wrist_goal_pos[env_ids, 0, 0] = center_x + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.offset_x).to(self.device)
        self.wrist_goal_pos[env_ids, 0, 1] = center_y + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.offset_y).to(self.device)
        self.wrist_goal_pos[env_ids, 0, 2] = center_z + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.offset_z).to(self.device)
        self.wrist_goal_pos[env_ids, 1, 0] = center_x + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.offset_x).to(self.device)
        self.wrist_goal_pos[env_ids, 1, 1] = center_y + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.offset_y).to(self.device)
        self.wrist_goal_pos[env_ids, 1, 2] = center_z + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.offset_z).to(self.device)


        humanoid_ids_int32 = self.humanoid_idxs[env_ids].to(torch.int32)
        ball_ids_int32 = self.ball_idxs[env_ids].to(torch.int32)
        small_box_ids_int32 = self.small_box_idxs[env_ids].to(torch.int32)

        ids = torch.cat(
            [
                humanoid_ids_int32,
                ball_ids_int32,
                small_box_ids_int32
            ]
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids), 
            len(ids)
        )

    def compute_observations(self):
        # Proprioception observations
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        proprioception_obs_buf = torch.cat((
            q,                                              # |num_actions|
            dq,                                             # |num_actions|
            self.actions,                                   # |num_actions|
            self.base_lin_vel * self.obs_scales.lin_vel,    # 3
            self.base_ang_vel * self.obs_scales.ang_vel,    # 3
            self.base_euler_xyz * self.obs_scales.quat,     # 3
        ), dim=-1)

        proprioception_privileged_obs_buf = torch.cat((
            q,                                              # |num_actions|
            dq,                                             # |num_actions|
            self.actions,                                   # |num_actions|
            self.base_lin_vel * self.obs_scales.lin_vel,    # 3
            self.base_ang_vel * self.obs_scales.ang_vel,    # 3
            self.base_euler_xyz * self.obs_scales.quat,     # 3
            self.rand_push_force[:, :2],                    # 2
            self.rand_push_torque,                          # 3
            self.env_frictions,                             # 1
            self.body_mass / 30.,                           # 1
            contact_mask                                    # 2
        ), dim=-1)

        # Task relevant observations (object, goal)
        humanoid_wrist_pos = self.rigid_state[:, self.wrist_indices, :3]
        ## Task reach
        wrist_goal_pos = self.wrist_goal_pos
        wrist_goal_pos_dist_obs = torch.flatten(humanoid_wrist_pos - wrist_goal_pos, start_dim=1)
        ## Task button
        humanoid_left_wrist_pos = humanoid_wrist_pos[:, 0]
        button_goal_pos = self.button_goal_pos
        wrist_button_dist_obs = humanoid_left_wrist_pos - button_goal_pos
        ## Task cabinet
        cabinet_pos = self.cabinet_root_states[:, :3]
        cabinet_dof_pos = self.cabinet_dof_state[:, :, 0]
        cabinet_dof_goal = self.cabinet_dof_goal
        wrist_cabinet_dist_obs = torch.flatten(humanoid_wrist_pos - cabinet_pos.unsqueeze(1), start_dim=1)
        cabinet_dof_pos_goal_dist_obs = cabinet_dof_pos - cabinet_dof_goal
        ## Task box
        small_box_pos = self.small_box_root_states[:, :3]
        wrist_small_box_dist_obs = torch.flatten(humanoid_wrist_pos - small_box_pos.unsqueeze(1), start_dim=1)
        small_box_goal_pos = self.small_box_goal_pos
        small_box_goal_dist_obs = small_box_pos - small_box_goal_pos
        ## Task ball
        humanoid_torso_pos = self.rigid_state[:, self.torso_indices, :3].squeeze(1)
        ball_pos = self.ball_root_states[:, :3]
        torso_ball_dist_obs = humanoid_torso_pos - ball_pos
        ball_goal_pos = self.ball_goal_pos
        ball_goal_dist_obs = ball_pos - ball_goal_pos
        task_obs_buf = torch.cat((
            ## Task reach
            wrist_goal_pos_dist_obs,                # 6
            ## Task button
            wrist_button_dist_obs,                  # 3
            ## Task cabinet
            wrist_cabinet_dist_obs,                 # 6
            cabinet_dof_pos_goal_dist_obs,          # 2
            ## Task box
            wrist_small_box_dist_obs,               # 6
            small_box_goal_dist_obs,                # 3
            ## Task ball
            torso_ball_dist_obs,                    # 3
            ball_goal_dist_obs,                     # 3
        ), dim=-1)

        task_privileged_obs_buf = torch.cat((
            torch.flatten(humanoid_wrist_pos, start_dim=1), # 6
            ## Task reach
            torch.flatten(wrist_goal_pos, start_dim=1),     # 6
            wrist_goal_pos_dist_obs,                        # 6
            ## Task button
            button_goal_pos,                                # 3
            wrist_button_dist_obs,                          # 3
            ## Task cabinet
            wrist_cabinet_dist_obs,                         # 6
            cabinet_dof_pos_goal_dist_obs,                  # 2
            ## Task box
            small_box_pos,                                  # 3
            wrist_small_box_dist_obs,                       # 6
            small_box_goal_pos,                             # 3
            small_box_goal_dist_obs,                        # 3
            ## Task ball
            humanoid_torso_pos,                             # 3
            ball_pos,                                       # 3
            torso_ball_dist_obs,                            # 3
            ball_goal_pos,                                  # 3
            ball_goal_dist_obs                              # 3
        ), dim=-1)

        # Phase observations
        task_one_hot = torch.zeros(self.num_envs, self.num_tasks, device=self.device)
        task_one_hot.scatter_(1, self.task_ids.view(-1, 1), 1.0)  # [num_envs, num_tasks]

        phase_obs_buf = task_one_hot
        phase_privileged_obs_buf = task_one_hot

        # Concat all observations
        obs_buf = torch.cat((
            proprioception_obs_buf,
            task_obs_buf,
            phase_obs_buf
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            proprioception_privileged_obs_buf,
            task_privileged_obs_buf,
            phase_privileged_obs_buf,
        ), dim=-1)

        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def step(self, actions):
        with torch.no_grad():
            # dynamic randomization
            delay = torch.rand((self.num_envs, 1), device=self.device)
            actions = (1 - delay) * actions.to(self.device) + delay * self.actions
            actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
            
            # changed version of super().step()
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
            # step physics and render each frame
            self.render()
            for _ in range(self.cfg.control.decimation):
                self.torques = self._compute_torques(self.actions).view(self.torques.shape) # [num_envs, num_actions]
                cabinet_force_buffer = torch.zeros((self.num_envs, self.cabinet_num_dofs), device=self.device)
                full_force_buffer = torch.cat((self.torques, cabinet_force_buffer), dim=1) # [num_envs, num_dofs + arti_obj_num_dofs]
                humanoid_ids_int32 = self.humanoid_idxs.to(dtype=torch.int32)
                self.gym.set_dof_actuation_force_tensor_indexed(self.sim, 
                                                                gymtorch.unwrap_tensor(full_force_buffer),
                                                                gymtorch.unwrap_tensor(humanoid_ids_int32), len(humanoid_ids_int32))

                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
            self.post_physics_step()

            # return clipped obs, clipped states (None), rewards, dones and infos
            clip_obs = self.cfg.normalization.clip_observations
            self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.task_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.humanoid_root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        alive_env_ids = (~self.reset_buf).nonzero(as_tuple=False).flatten()
        self.check_switch(alive_env_ids)
        switch_env_ids = self.switch_buf.nonzero(as_tuple=False).flatten()
        self.switch_idx(switch_env_ids)
        self.reset_idx(reset_env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.humanoid_root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        # if the ball hits the goal, reset the env
        ball_pos = self.ball_root_states[:, :3]
        goal_pos = self.ball_goal_pos
        ball_goal_diff = ball_pos - goal_pos # [envs, 3]
        ball_goal_dist = torch.norm(ball_goal_diff, dim=1)
        self.reset_buf |= ball_goal_dist < self.cfg.commands.ranges.threshold

    def check_switch(self, env_ids):
        timeout = (self.task_length_buf[env_ids] >= self.max_task_length[env_ids])
        is_last = (self.task_ptr[env_ids] >= self.num_chain - 1)

        self.switch_buf[env_ids] = timeout & (~is_last)

        self.reset_buf[env_ids] |= timeout & is_last

    def switch_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        new_task_ptr = self.task_ptr[env_ids] + 1

        is_last = new_task_ptr >= self.num_chain
        if is_last.any():
            overflow_env_ids = env_ids[is_last]
            self.reset_buf[overflow_env_ids] = True

            env_ids = env_ids[~is_last]
            if len(env_ids) == 0:
                return
            
            new_task_ptr = new_task_ptr[~is_last]

        self.task_ptr[env_ids] = new_task_ptr
        self.task_ids[env_ids] = self.task_chain[new_task_ptr]
        self.task_length_buf[env_ids] = 0
        self.max_task_length[env_ids] = self.chain_max_task_length[new_task_ptr]
        self.switch_buf[env_ids] = False

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.task_ptr[env_ids] = 0
        self.task_ids[env_ids] = self.task_chain[0]
        self.task_length_buf[env_ids] = 0
        self.max_task_length[env_ids] = self.chain_max_task_length[0]

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #
    def _task_mask(self, task_id):
        return (self.task_ids == task_id).float()
    
    # Task rewards
    ## Task reach
    ### Main goal
    def _reward_wrist_goal_distance(self):
        wrist_pos = self.rigid_state[:, self.wrist_indices, :3]
        wrist_goal_pos = self.wrist_goal_pos
        wrist_goal_distance = torch.flatten(wrist_pos - wrist_goal_pos, start_dim=1)
        wrist_goal_error = torch.mean(torch.abs(wrist_goal_distance), dim=1)
        reward = torch.exp(-4 * wrist_goal_error)
        mask = self._task_mask(self.TASK_REACH)
        return mask * reward

    ## Task button
    ### Main goal
    def _reward_wrist_button_distance(self):
        left_wrist_pos = self.rigid_state[:, self.wrist_indices[0], :3]
        button_goal_pos = self.button_goal_pos
        wrist_button_distance = left_wrist_pos - button_goal_pos
        wrist_button_error = torch.mean(torch.abs(wrist_button_distance), dim=1)
        reward = torch.exp(-4 * wrist_button_error)
        mask = self._task_mask(self.TASK_BUTTON)
        return mask * reward

    ## Task cabinet
    def _reward_wrist_cabinet_distance(self):
        wrist_pos = self.rigid_state[:, self.wrist_indices, :3]
        cabinet_pos = self.cabinet_root_states[:, :3]
        wrist_cabinet_distance = torch.flatten(wrist_pos - cabinet_pos.unsqueeze(1), start_dim=1)
        wrist_cabinet_error = torch.mean(torch.abs(wrist_cabinet_distance), dim=1)
        reward = torch.exp(-4 * wrist_cabinet_error)
        mask = self._task_mask(self.TASK_CABINET)
        return mask * reward

    ### Main goal
    def _reward_cabinet_goal_distance(self):
        cabinet_dof_state = self.cabinet_dof_state[:, :, 0]
        cabinet_dof_state_goal = self.cabinet_dof_goal
        cabinet_goal_distance = cabinet_dof_state - cabinet_dof_state_goal
        cabinet_goal_error = torch.mean(torch.abs(cabinet_goal_distance), dim=1)
        reward = torch.exp(-4 * cabinet_goal_error)
        mask = self._task_mask(self.TASK_CABINET)
        return mask * reward

    ## Task box
    def _reward_wrist_small_box_distance(self):
        wrist_pos = self.rigid_state[:, self.wrist_indices, :3]
        small_box_pos = self.small_box_root_states[:, :3]
        wrist_small_box_distance = torch.flatten(wrist_pos - small_box_pos.unsqueeze(1), start_dim=1)
        wrist_small_box_error = torch.mean(torch.abs(wrist_small_box_distance), dim=1)
        reward = torch.exp(-4 * wrist_small_box_error)
        mask = self._task_mask(self.TASK_BOX)
        return mask * reward

    ### Main goal
    def _reward_small_box_goal_distance(self):
        small_box_pos = self.small_box_root_states[:, :3]
        small_box_goal_pos = self.small_box_goal_pos
        small_box_goal_distance = small_box_pos - small_box_goal_pos
        small_box_goal_error = torch.mean(torch.abs(small_box_goal_distance), dim=1)
        reward = torch.exp(-4 * small_box_goal_error) 
        mask = self._task_mask(self.TASK_BOX)
        return mask * reward

    ## Task ball
    def _reward_torso_ball_distance(self):
        torso_pos = self.rigid_state[:, self.torso_indices, :3].squeeze(1)
        ball_pos = self.ball_root_states[:, :3]
        torso_ball_distance = (torso_pos - ball_pos)[:, :2]
        torso_ball_error = torch.mean(torch.abs(torso_ball_distance), dim=1)
        reward = torch.exp(-4 * torso_ball_error)
        mask = self._task_mask(self.TASK_BALL)
        return mask * reward

    ### Main goal
    def _reward_ball_goal_distance(self):
        ball_pos = self.ball_root_states[:, :3]
        ball_goal_pos = self.ball_goal_pos
        ball_goal_distance = ball_pos - ball_goal_pos
        ball_goal_error = torch.mean(torch.abs(ball_goal_distance), dim=1)
        reward = torch.exp(-4 * ball_goal_error)
        mask = self._task_mask(self.TASK_BALL)
        return mask * reward

    # Base rewards
    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.
    
    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, foot_dist
    
    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.rigid_state[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, knee_dist
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
# ==== From HumanPlus ==== #
    ## _reward_lin_vel_z, _reward_ang_vel_xy, _reward_action_rate, _reward_termination, _reward_dof_pos_limits, _reward_dof_vel_limits, _reward_torque_limits, **_reward_stumble**, **_reward_stand_still**, _reward_target_jt (joint_pos)
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_target_jt(self):
        # # Penalize distance to target joint angles
        # target_jt_error = torch.mean(torch.abs(self.dof_pos - self.target_jt), dim=1)
        # return torch.exp(-4 * target_jt_error), target_jt_error
        return 0