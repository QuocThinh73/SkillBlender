from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from .h1_multitask_config import H1MultitaskCfg

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot, LEGGED_GYM_ROOT_DIR, get_euler_xyz_tensor, quat_rotate_inverse

import os
import numpy as np


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

        # Goals
        ## Task reach
        self.wrist_goal_pos = torch.zeros(self.num_envs, 2, 7, device=self.device)
        ## Task button
        self.button_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        ## Task cabinet
        self.cabinet_dof_goal = 0
        ## Task box
        self.small_box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        ## Task ball
        self.ball_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _push_robots(self):
        pass

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
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

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

            actor_handle = self.gym.create_actor(env_handle, humanoid_asset, humanoid_pose, "h1", i, 0, 0)
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
            ball_pose.p.x = pos[0].item() + 2
            ball_pose.p.y = pos[1].item()
            ball_pose.p.z = pos[2].item() + 0.5 * ball_size
            ball_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-np.pi, np.pi))
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 0)
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
            small_box_pose.p.z = table_pose.p.z
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
        if len(env_ids) == 0:
            return

        # Reset humanoid root states
        self.humanoid_root_states[env_ids] = self.humanoid_base_init_state
        self.humanoid_root_states[env_ids, :3] += self.env_origins[env_ids]
        # Reset ball root states
        self.ball_root_states[env_ids, :3] = self.env_origins[env_ids]
        self.ball_root_states[env_ids, 0] += 2.0
        self.ball_root_states[env_ids, 1] += 0.0
        self.ball_root_states[env_ids, 2] += 0.5 * self.cfg.asset.ball_size
        # Reset small box root states
        self.small_box_root_states[env_ids, :3] = self.env_origins[env_ids]
        self.small_box_root_states[env_ids, 0] += self.cfg.asset.table_offsets[0]
        self.small_box_root_states[env_ids, 1] += self.cfg.asset.table_offsets[1]
        self.small_box_root_states[env_ids, 2] += self.cfg.asset.table_offsets[2]

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

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def compute_observations(self):
        # Proprioception observations
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        proprioception_obs_buf = torch.cat((
            q,                                              # |num_actions|
            dq,                                             # |num_actions|
            self.actions,                                   # |num_actions|
            self.base_lin_vel * self.obs_scales.lin_vel,    # 3
            self.base_ang_vel * self.obs_scales.ang_vel,    # 3
            self.base_euler_xyz * self.obs_scales.quat,     # 3
        ), dim=-1)

        # Task relevant observations (object, goal)
        humanoid_wrist = self.rigid_state[:, self.wrist_indices, :7]
        humanoid_wrist_pos = humanoid_wrist[:, :, :3]
        ## Task reach
        wrist_goal_pos = self.wrist_goal_pos
        wrist_goal_pos_dist_obs = torch.flatten(humanoid_wrist_pos - wrist_goal_pos, start_dim=1)
        ## Task button
        humanoid_left_wrist_pos = humanoid_wrist[:, 0, :3]
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
        small_box_goal_dist_obs = small_box_pos - self.small_box_goal_pos
        ## Task ball
        humanoid_torso_pos = self.rigid_state[:, self.torso_indices, :3].squeeze(1)
        ball_pos = self.ball_root_states[:, :3]
        torso_ball_dist_obs = humanoid_torso_pos - ball_pos
        ball_goal_dist_obs = ball_pos - self.ball_goal_pos

        task_obs_buf = torch.cat((
            ## Task reach
            wrist_goal_pos_dist_obs,                # 14
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

        # Phase observations

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
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.humanoid_root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

# ================================================ Rewards ================================================== #
