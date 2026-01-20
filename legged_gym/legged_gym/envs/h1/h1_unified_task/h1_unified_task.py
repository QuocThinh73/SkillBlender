import torch
import random
import numpy as np
import math
import os
import torch.nn.functional as F

from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot import LeggedRobot, get_euler_xyz_tensor
from isaacgym import gymapi, gymtorch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import XBotTerrain
from collections import deque
from legged_gym.utils.human import sample_int_from_float, sample_rp
from copy import deepcopy


class H1UnifiedTask(LeggedRobot):
    '''
    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg

        self.hidden_z = cfg.asset.hidden_z

        self.num_tasks = self.cfg.task.num_tasks
        self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_ids = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)

        # Task ball
        self.ori_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        door_z_offsets = [offset[2] for offset in self.cfg.asset.door_offsets]
        self.door_z_offsets = torch.tensor(door_z_offsets, device=self.device)
        self.num_door_parts = len(self.cfg.asset.door_offsets)

        # Task box
        self.box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task button
        self.button_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task cabinet

        # Task carry
        self.box_carry_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task lift
        self.box_lift_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task reach

        # Task transfer
        self.box_transfer_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.gym.simulate(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._init_target_wp()
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
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

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase
    
    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def _init_target_wp(self):
        self.ori_wrist_pos = self.rigid_state[:, self.wrist_indices, :7].clone() # [num_envs, 2, 7], two hands
        self.target_wp, self.num_pairs, self.num_wp = sample_rp(self.device, num_points=2000000, num_wp=10, ranges=self.cfg.commands.ranges) # relative, self.target_wp.shape=[num_pairs, num_wp, 2, 7]
        self.target_wp_i = torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device) # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / self.cfg.human.freq
        self.target_wp_update_steps = self.target_wp_dt / self.dt # not necessary integer
        assert self.dt <= self.target_wp_dt, f"self.dt {self.dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
        
        self.ref_dof_pos = None
        self.ref_wrist_pos = None
        self.ref_action = self.default_dof_pos
        self.delayed_obs_target_wp = None
        self.delayed_obs_target_wp_steps = self.cfg.human.delay / self.target_wp_dt
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.update_target_wp(torch.tensor([], dtype=torch.long, device=self.device))


    def update_target_wp(self, reset_env_ids):
        # self.target_wp_i specifies which seq to use for each env, and self.target_wp_j specifies the timestep in the seq
        self.ref_wrist_pos = self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_wrist_pos # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_wp_update_steps_int == 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        if self.cfg.human.resample_on_env_reset:
            self.target_wp_j[reset_env_ids] = 0
            resample_i[reset_env_ids] = True
        self.target_wp_i = torch.where(resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i)


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = XBotTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
        
    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # Create assets
        ## Task ball
        ### Door asset
        door_assets = []
        door_pose = gymapi.Transform()
        for dim in self.cfg.asset.door_dims:
            door_dims = gymapi.Vec3(*dim)
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            door_asset = self.gym.create_box(self.sim, door_dims.x, door_dims.y, door_dims.z, asset_options)
            door_assets.append(door_asset)
        self.door_idxs = []
        
        ### Ball asset
        ball_size = self.cfg.asset.ball_size
        asset_options = gymapi.AssetOptions()
        ball_asset = self.gym.create_sphere(self.sim, ball_size, asset_options)
        ball_pose = gymapi.Transform()
        self.ball_idxs = []
        
        ## Task box
        ### Table asset
        table_dims = gymapi.Vec3(*self.cfg.asset.table_dims)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        self.table_idxs = []

        ### Box asset
        box_size = self.cfg.asset.box_size
        asset_options = gymapi.AssetOptions()
        box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        box_pose = gymapi.Transform()
        self.box_idxs = []
            
        ## Task Button
        ### Wall asset
        wall_dims = gymapi.Vec3(*self.cfg.asset.wall_dims)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        wall_asset = self.gym.create_box(self.sim, wall_dims.x, wall_dims.y, wall_dims.z, asset_options)
        wall_pose = gymapi.Transform()
        self.wall_idxs = []

        ## Task Cabinet
        ### Cabinet asset
        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.use_mesh_materials = True
        arti_obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        arti_obj_asset_options.override_inertia = True
        arti_obj_asset_options.override_com = True
        arti_obj_asset_options.fix_base_link = True
        arti_obj_asset_options.disable_gravity = True
        arti_obj_asset = self.gym.load_asset(self.sim, self.cfg.asset.gapartnet_root, f"{self.cfg.asset.gapartnet_id}/mobility_annotation_gapartnet.urdf", arti_obj_asset_options)
        self.arti_obj_num_dofs = self.gym.get_asset_dof_count(arti_obj_asset)
        print("=====> arti_obj_num_dofs:", self.arti_obj_num_dofs)

        ### Configure object dofs
        arti_obj_dof_props = self.gym.get_asset_dof_properties(arti_obj_asset)
        arti_obj_default_dof_pos = np.zeros(self.arti_obj_num_dofs, dtype=np.float32)
        arti_obj_default_dof_pos[:] = self.cfg.asset.arti_obj_dof_default
        arti_obj_default_dof_state = np.zeros(self.arti_obj_num_dofs, gymapi.DofState.dtype)
        arti_obj_default_dof_state["pos"] = arti_obj_default_dof_pos
        arti_obj_dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE) # NO DOF_MODE_POS
        arti_obj_dof_props["stiffness"].fill(0) # how fast the arti obj gonna move
        arti_obj_dof_props["damping"].fill(5) # large damping to prevent oscillation
        arti_obj_dof_props["friction"].fill(0)
        arti_obj_pose = gymapi.Transform()
        self.arti_obj_idxs = []

        ## Task Carry
        ### Box carry asset
        box_carry_size = self.cfg.asset.box_carry_size
        asset_options = gymapi.AssetOptions()
        box_carry_asset = self.gym.create_box(self.sim, box_carry_size[0], box_carry_size[1], box_carry_size[2], asset_options)
        box_carry_pose = gymapi.Transform()
        self.box_carry_idxs = []

        ## Task Lift
        ### Box lift asset
        box_lift_size = self.cfg.asset.box_lift_size
        asset_options = gymapi.AssetOptions()
        # asset_options.density = self.cfg.asset.density * 0.5 # no need to do this, will change mass below
        # add rigid shape properties if needed (e.g. friction)
        box_lift_asset = self.gym.create_box(self.sim, box_lift_size[0], box_lift_size[1], box_lift_size[2], asset_options)
        box_lift_pose = gymapi.Transform()
        self.box_lift_idxs = []
        
        ## Task Transfer
        ### Front table asset
        front_table_dims = gymapi.Vec3(*self.cfg.asset.front_table_dims)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        front_table_asset = self.gym.create_box(self.sim, front_table_dims.x, front_table_dims.y, front_table_dims.z, asset_options)
        front_table_pose = gymapi.Transform()
        self.front_table_idxs = []
        ### Back table asset
        back_table_dims = gymapi.Vec3(*self.cfg.asset.back_table_dims)
        back_table_asset = self.gym.create_box(self.sim, back_table_dims.x, back_table_dims.y, back_table_dims.z, asset_options)
        back_table_pose = gymapi.Transform()
        self.back_table_idxs = []
        ### Box transfer asset
        box_transfer_size = self.cfg.asset.box_transfer_size
        asset_options = gymapi.AssetOptions()
        box_transfer_asset = self.gym.create_box(self.sim, box_transfer_size, box_transfer_size, box_transfer_size, asset_options)
        box_transfer_pose = gymapi.Transform()
        self.box_transfer_idxs = []
        
        # Create actors
        self.humanoid_idxs = []
        for i in range(self.num_envs):
            ## Create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.humanoid_idxs.append(self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))

            ## Task ball
            ### Add door
            env_door_idxs = []
            for door_i, door_asset in enumerate(door_assets):
                door_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.door_offsets[door_i], device=self.device)))
                door_handle = self.gym.create_actor(env_handle, door_asset, door_pose, f"door_{door_i}", i, 0)
                env_door_idxs.append(self.gym.get_actor_index(env_handle, door_handle, gymapi.DOMAIN_SIM))
            self.door_idxs.append(env_door_idxs)
            ### Add ball
            ball_pose.p.x = pos[0].item() + np.random.uniform(*self.cfg.asset.ball_range_x)
            ball_pose.p.y = pos[1].item() + np.random.uniform(*self.cfg.asset.ball_range_y)
            ball_pose.p.z = pos[2].item() + 0.5 * ball_size
            ball_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 0)
            ### Change ball actor properties
            ball_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            for prop in ball_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.ball_range_mass) # change mass here!
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.ball_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))

            ## Task box
            ### Add table
            table_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.table_offset, device=self.device)))
            table_handle = self.gym.create_actor(env_handle, table_asset, table_pose, "table", i, 0)
            self.table_idxs.append(self.gym.get_actor_index(env_handle, table_handle, gymapi.DOMAIN_SIM))
            ### Add box
            box_pose.p.x = table_pose.p.x + np.random.uniform(*self.cfg.asset.box_range_x)
            box_pose.p.y = table_pose.p.y + np.random.uniform(*self.cfg.asset.box_range_y)
            box_pose.p.z = table_pose.p.z + 0.5 * self.cfg.asset.table_dims[2] + 0.5 * box_size
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_handle = self.gym.create_actor(env_handle, box_asset, box_pose, "box", i, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.box_idxs.append(self.gym.get_actor_index(env_handle, box_handle, gymapi.DOMAIN_SIM))

            ## Task button
            ### Add wall
            wall_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.wall_offset, device=self.device)))
            wall_handle = self.gym.create_actor(env_handle, wall_asset, wall_pose, "wall", i, 0)
            self.wall_idxs.append(self.gym.get_actor_index(env_handle, wall_handle, gymapi.DOMAIN_SIM))
            
            ## Task cabinet
            ### Add cabinet
            arti_obj_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.arti_obj_offset, device=self.device)))
            arti_obj_handle = self.gym.create_actor(env_handle, arti_obj_asset, arti_obj_pose, "cabinet", i, 0)
            self.gym.set_actor_dof_properties(env_handle, arti_obj_handle, arti_obj_dof_props)
            self.gym.set_actor_dof_states(env_handle, arti_obj_handle, arti_obj_default_dof_state, gymapi.STATE_ALL)
            # self.gym.set_actor_dof_position_targets(env_handle, arti_obj_handle, arti_obj_default_dof_pos) # not sure whether it's useful
            self.gym.set_actor_scale(env_handle, arti_obj_handle, self.cfg.asset.arti_obj_scale)
            self.arti_obj_idxs.append(self.gym.get_actor_index(env_handle, arti_obj_handle, gymapi.DOMAIN_SIM))

            ## Task Carry
            ### Add box carry
            box_carry_pose.p = gymapi.Vec3(*pos[:3])
            box_carry_pose.p.x += self.cfg.asset.box_carry_offset_xy[0] + np.random.uniform(*self.cfg.asset.box_carry_range_x)
            box_carry_pose.p.y += self.cfg.asset.box_carry_offset_xy[1] + np.random.uniform(*self.cfg.asset.box_carry_range_y)
            box_carry_pose.p.z = 0.5 * box_carry_size[2]
            # box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_carry_handle = self.gym.create_actor(env_handle, box_carry_asset, box_carry_pose, "box_carry", i, 0)
            ### change box actor properties
            box_carry_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_carry_handle)
            for prop in box_carry_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.box_carry_range_mass) # change mass here!
            self.gym.set_actor_rigid_body_properties(env_handle, box_carry_handle, box_carry_rigid_body_props, recomputeInertia=True)
            box_carry_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, box_carry_handle)
            for prop in box_carry_rigid_shape_props:
                prop.friction = 5. # change friction here!
            self.gym.set_actor_rigid_shape_properties(env_handle, box_carry_handle, box_carry_rigid_shape_props)
            ###
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, box_carry_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.box_carry_idxs.append(self.gym.get_actor_index(env_handle, box_carry_handle, gymapi.DOMAIN_SIM))

            ## Task Lift
            ### Add box lift
            box_lift_pose.p = gymapi.Vec3(*pos[:3])
            box_lift_pose.p.x += self.cfg.asset.box_lift_offset_xy[0] + np.random.uniform(*self.cfg.asset.box_lift_range_x)
            box_lift_pose.p.y += self.cfg.asset.box_lift_offset_xy[1] + np.random.uniform(*self.cfg.asset.box_lift_range_y)
            box_lift_pose.p.z = 0.5 * box_lift_size[2]
            # box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_lift_handle = self.gym.create_actor(env_handle, box_lift_asset, box_lift_pose, "box_lift", i, 0)
            ### change box actor properties
            box_lift_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_lift_handle)
            for prop in box_lift_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.box_lift_range_mass) # change mass here!
            self.gym.set_actor_rigid_body_properties(env_handle, box_lift_handle, box_lift_rigid_body_props, recomputeInertia=True)
            box_lift_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, box_lift_handle)
            for prop in box_lift_rigid_shape_props:
                prop.friction = 5. # change friction here!
            self.gym.set_actor_rigid_shape_properties(env_handle, box_lift_handle, box_lift_rigid_shape_props)
            ###
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, box_lift_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.box_lift_idxs.append(self.gym.get_actor_index(env_handle, box_lift_handle, gymapi.DOMAIN_SIM))
            
            ## Task Transfer
            ### Add front table
            front_table_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.front_table_offset, device=self.device)))
            front_table_handle = self.gym.create_actor(env_handle, front_table_asset, front_table_pose, "front_table", i, 0)
            self.front_table_idxs.append(self.gym.get_actor_index(env_handle, front_table_handle, gymapi.DOMAIN_SIM))
            ### Add back table
            back_table_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.back_table_offset, device=self.device)))
            back_table_handle = self.gym.create_actor(env_handle, back_table_asset, back_table_pose, "back_table", i, 0)
            self.back_table_idxs.append(self.gym.get_actor_index(env_handle, back_table_handle, gymapi.DOMAIN_SIM))
            ### Add box transfer
            box_transfer_pose.p.x = front_table_pose.p.x + np.random.uniform(*self.cfg.asset.box_transfer_range_x)
            box_transfer_pose.p.y = front_table_pose.p.y + np.random.uniform(*self.cfg.asset.box_transfer_range_y)
            box_transfer_pose.p.z = front_table_pose.p.z + 0.5 * self.cfg.asset.front_table_dims[2] + 0.5 * self.cfg.asset.box_transfer_size
            # box_transfer_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_transfer_handle = self.gym.create_actor(env_handle, box_transfer_asset, box_transfer_pose, "box_transfer", i, 0)
            ### change box actor properties
            box_transfer_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_transfer_handle)
            for prop in box_transfer_rigid_body_props:
                prop.mass = self.cfg.asset.box_transfer_mass # change mass here!
            ###
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, box_transfer_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.box_transfer_idxs.append(self.gym.get_actor_index(env_handle, box_transfer_handle, gymapi.DOMAIN_SIM))


        self._create_sensors_all()
        self.humanoid_idxs = torch.tensor(self.humanoid_idxs, device=self.device)

        ## Task ball
        self.ball_idxs = torch.tensor(self.ball_idxs, device=self.device)
        self.door_idxs = torch.tensor(self.door_idxs, device=self.device)
        ## Task box
        self.table_idxs = torch.tensor(self.table_idxs, device=self.device)
        self.box_idxs = torch.tensor(self.box_idxs, device=self.device)
        ## Task button
        self.wall_idxs = torch.tensor(self.wall_idxs, device=self.device)
        ## Task cabinet
        self.arti_obj_idxs = torch.tensor(self.arti_obj_idxs, device=self.device)
        ## Task carry
        self.box_carry_idxs = torch.tensor(self.box_carry_idxs, device=self.device)
        ## Task lift
        self.box_lift_idxs = torch.tensor(self.box_lift_idxs, device=self.device)
        ## Task transfer
        self.front_table_idxs = torch.tensor(self.front_table_idxs, device=self.device)
        self.back_table_idxs = torch.tensor(self.back_table_idxs, device=self.device)
        self.box_transfer_idxs = torch.tensor(self.box_transfer_idxs, device=self.device)


        ### Common body parts
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

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
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self._init_visual_buffers()
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.humanoid_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.humanoid_idxs[0]]
        
        # Task ball
        self.ball_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.ball_idxs[0]]
        # Task box
        self.table_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.table_idxs[0]]
        self.box_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.box_idxs[0]]
        # Task button
        self.wall_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.wall_idxs[0]]
        # Task cabinet
        self.arti_obj_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.arti_obj_idxs[0]]
        # Task carry
        self.box_carry_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.box_carry_idxs[0]]
        # Task lift
        self.box_lift_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.box_lift_idxs[0]]
        # Task transfer
        self.front_table_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.front_table_idxs[0]]
        self.back_table_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.back_table_idxs[0]]
        self.box_transfer_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.box_transfer_idxs[0]]


        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Task cabinet
        self.humanoid_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof]
        self.arti_obj_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:]

        self.dof_pos = self.humanoid_dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.humanoid_dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.humanoid_root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
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
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(name)
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
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
            
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        ## Task cabinet
        self._reset_task_cabinet(env_ids)

        humanoid_ids_int32 = self.humanoid_idxs[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(humanoid_ids_int32), len(humanoid_ids_int32))
        
        ## Task cabinet
        arti_obj_ids_int32 = self.arti_obj_idxs[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(arti_obj_ids_int32), len(arti_obj_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.humanoid_root_states[env_ids] = self.base_init_state
            self.humanoid_root_states[env_ids, :3] += self.env_origins[env_ids]
            self.humanoid_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.humanoid_root_states[env_ids] = self.base_init_state
            self.humanoid_root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.humanoid_root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.humanoid_root_states[env_ids, 7:13] = 0
            self.humanoid_root_states[env_ids, 2] += 1.8
        
        ## Task ball
        self._reset_task_ball(env_ids)
        ## Task box
        self._reset_task_box(env_ids)
        ## Task button
        self._reset_task_button(env_ids)
        ## Task cabinet
        self._reset_task_cabinet(env_ids)
        ## Task carry
        self._reset_task_carry(env_ids)
        ## Task lift
        self._reset_task_lift(env_ids)
        ## Task transfer
        self._reset_task_transfer(env_ids)
        
        humanoid_ids_int32 = self.humanoid_idxs[env_ids].to(dtype=torch.int32)
                
        ## Task ball
        ball_ids_int32 = self.ball_idxs[env_ids].to(dtype=torch.int32)
        door_ids_int32 = self.door_idxs[env_ids].flatten().to(dtype=torch.int32)
        ## Task box
        box_ids_int32 = self.box_idxs[env_ids].to(dtype=torch.int32)
        table_ids_int32 = self.table_idxs[env_ids].to(dtype=torch.int32)
        ## Task button
        wall_ids_int32 = self.wall_idxs[env_ids].to(dtype=torch.int32)
        ## Task cabinet
        arti_obj_ids_int32 = self.arti_obj_idxs[env_ids].to(dtype=torch.int32)
        ## Task carry
        box_carry_ids_int32 = self.box_carry_idxs[env_ids].to(dtype=torch.int32)
        ## Task lift
        box_lift_ids_int32 = self.box_lift_idxs[env_ids].to(dtype=torch.int32)
        ## Task transfer
        front_table_ids_int32 = self.front_table_idxs[env_ids].to(dtype=torch.int32)
        back_table_ids_int32 = self.back_table_idxs[env_ids].to(dtype=torch.int32)
        box_transfer_ids_int32 = self.box_transfer_idxs[env_ids].to(dtype=torch.int32)
        
        all_actor_indices = torch.cat([
            humanoid_ids_int32,
            ball_ids_int32,
            door_ids_int32,
            box_ids_int32,
            table_ids_int32,
            wall_ids_int32,
            arti_obj_ids_int32,
            box_carry_ids_int32,
            box_lift_ids_int32,
            front_table_ids_int32,
            back_table_ids_int32,
            box_transfer_ids_int32
        ])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(all_actor_indices),
            len(all_actor_indices)
        )

    def _reset_task_ball(self, env_ids):
        task_ball_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_BALL)

        active_ids = env_ids[task_ball_envs]
        inactive_ids = env_ids[~task_ball_envs]

        if len(active_ids) > 0:
            pos = self.env_origins[active_ids].clone()

            # Reset ball
            self.ori_ball_pos[active_ids, 0] = pos[:, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.ball_range_x).to(self.device)
            self.ori_ball_pos[active_ids, 1] = pos[:, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.ball_range_y).to(self.device)
            self.ori_ball_pos[active_ids, 2] = 0.5 * self.cfg.asset.ball_size
            self.ball_root_states[active_ids, :3] = self.ori_ball_pos[active_ids].clone()
            self.ball_root_states[active_ids, 3] = 1
            self.ball_root_states[active_ids, 4:] = 0

            # Reset door
            active_door_indices = self.door_idxs[active_ids].flatten()
            target_z_offsets = self.door_z_offsets.repeat(len(active_ids))
            env_base_z = pos[:, 2].repeat_interleave(self.num_door_parts)
            final_z = env_base_z + target_z_offsets
            self.root_states[active_door_indices, 2] = final_z
            self.root_states[active_door_indices, 7:13] = 0

            # Reset goal
            self.goal_pos[active_ids, 0] = pos[:, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.goal_x).to(self.device)
            self.goal_pos[active_ids, 1] = pos[:, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.goal_y).to(self.device)
            self.goal_pos[active_ids, 2] = torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.goal_z).to(self.device)

        if len(inactive_ids) > 0:
            self.ball_root_states[inactive_ids, 2] = self.hidden_z
            self.ball_root_states[inactive_ids, 7:13] = 0

            inactive_door_indices = self.door_idxs[inactive_ids].flatten()
            self.root_states[inactive_door_indices, 2] = self.hidden_z
            self.root_states[inactive_door_indices, 7:13] = 0

    def _reset_task_box(self, env_ids):
        task_box_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_BOX)

        active_ids = env_ids[task_box_envs]
        inactive_ids = env_ids[~task_box_envs]

        if len(active_ids) > 0:
            pos = self.env_origins[active_ids].clone()

            # Reset table
            self.table_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.table_offset[0]
            self.table_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.table_offset[1]
            self.table_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.table_offset[2]
            self.table_root_states[active_ids, 7:13] = 0

            # Reset box
            self.box_root_states[active_ids, 0] = self.table_root_states[active_ids, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_range_x).to(self.device)
            self.box_root_states[active_ids, 1] = self.table_root_states[active_ids, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_range_y).to(self.device)
            self.box_root_states[active_ids, 2] = self.table_root_states[active_ids, 2] + 0.5 * self.cfg.asset.table_dims[2] + 0.5 * self.cfg.asset.box_size
            self.box_root_states[active_ids, 3] = 1
            self.box_root_states[active_ids, 4:] = 0

            # Reset goal
            self.box_goal_pos[active_ids, 0] = self.box_root_states[active_ids, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.box_pos_x).to(self.device)
            self.box_goal_pos[active_ids, 1] = self.box_root_states[active_ids, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.box_pos_y).to(self.device)
            self.box_goal_pos[active_ids, 2] = self.box_root_states[active_ids, 2].clone()

        if len(inactive_ids) > 0:
            self.table_root_states[inactive_ids, 2] = self.hidden_z
            self.table_root_states[inactive_ids, 7:13] = 0

            self.box_root_states[inactive_ids, 2] = self.hidden_z
            self.box_root_states[inactive_ids, 7:13] = 0
        
    def _reset_task_button(self, env_ids):
        task_button_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_BUTTON)

        active_ids = env_ids[task_button_envs]
        inactive_ids = env_ids[~task_button_envs]

        if len(active_ids) > 0:
            pos = self.env_origins[active_ids].clone()

            # Reset wall
            self.wall_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.wall_offset[0]
            self.wall_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.wall_offset[1]
            self.wall_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.wall_offset[2]
            self.wall_root_states[active_ids, 7:13] = 0

            # Reset goal
            self.button_goal_pos[active_ids, 0] = self.wall_root_states[active_ids, 0].clone()
            self.button_goal_pos[active_ids, 1] = self.wall_root_states[active_ids, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.button_pos_y).to(self.device)
            self.button_goal_pos[active_ids, 2] = self.cfg.asset.button_ori_z + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.button_pos_z).to(self.device)

        if len(inactive_ids) > 0:
            self.wall_root_states[inactive_ids, 2] = self.hidden_z
            self.wall_root_states[inactive_ids, 7:13] = 0
       
    def _reset_task_cabinet(self, env_ids):
        self.arti_obj_dof_state[env_ids, :, 0] = self.cfg.asset.arti_obj_dof_default 
        self.arti_obj_dof_state[env_ids, :, 1] = 0 
        self.arti_obj_dof_goal = 0

        task_cabinet_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_CABINET)

        active_ids = env_ids[task_cabinet_envs]
        inactive_ids = env_ids[~task_cabinet_envs]

        if len(active_ids) > 0:
            active_indices = self.arti_obj_idxs[active_ids]
            self.root_states[active_indices, 2] = self.cfg.asset.arti_obj_offset[2]
            self.root_states[active_indices, 7:13] = 0

        if len(inactive_ids) > 0:
            inactive_indices = self.arti_obj_idxs[inactive_ids]
            self.root_states[inactive_indices, 2] = self.hidden_z
            self.root_states[inactive_indices, 7:13] = 0

    def _reset_task_carry(self, env_ids):
        task_carry_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_CARRY)

        active_ids = env_ids[task_carry_envs]
        inactive_ids = env_ids[~task_carry_envs]

        if len(active_ids) > 0:
            pos = self.env_origins[active_ids].clone()

            # Reset box carry
            self.box_carry_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.box_carry_offset_xy[0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_carry_range_x).to(self.device)
            self.box_carry_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.box_carry_offset_xy[1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_carry_range_y).to(self.device)
            self.box_carry_root_states[active_ids, 2] = 0.5 * self.cfg.asset.box_carry_size[2]
            self.box_carry_root_states[active_ids, 3] = 1
            self.box_carry_root_states[active_ids, 4:] = 0

            # Reset goal
            self.box_carry_goal_pos[active_ids, 0] = pos[:, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.box_carry_pos_x).to(self.device)
            self.box_carry_goal_pos[active_ids, 1] = pos[:, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.box_carry_pos_y).to(self.device)
            self.box_carry_goal_pos[active_ids, 2] = self.box_carry_root_states[active_ids, 2].clone()
        
        if len(inactive_ids) > 0:
            self.box_carry_root_states[inactive_ids, 2] = self.hidden_z
            self.box_carry_root_states[inactive_ids, 7:13] = 0

    def _reset_task_lift(self, env_ids):
        task_lift_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_LIFT)

        active_ids = env_ids[task_lift_envs]
        inactive_ids = env_ids[~task_lift_envs]

        if len(active_ids) > 0:
            pos = self.env_origins[active_ids].clone()

            # Reset box lift
            self.box_lift_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.box_lift_offset_xy[0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_lift_range_x).to(self.device)
            self.box_lift_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.box_lift_offset_xy[1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_lift_range_y).to(self.device)
            self.box_lift_root_states[active_ids, 2] = 0.5 * self.cfg.asset.box_lift_size[2]
            self.box_lift_root_states[active_ids, 3] = 1
            self.box_lift_root_states[active_ids, 4:] = 0

            # Reset goal
            self.box_lift_goal_pos[active_ids, 0] = self.box_lift_root_states[active_ids, 0].clone()
            self.box_lift_goal_pos[active_ids, 1] = self.box_lift_root_states[active_ids, 1].clone()
            self.box_lift_goal_pos[active_ids, 2] = self.box_lift_root_states[active_ids, 2] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.commands.ranges.box_lift_pos_z).to(self.device)

        if len(inactive_ids) > 0:
            self.box_lift_root_states[inactive_ids, 2] = self.hidden_z
            self.box_lift_root_states[inactive_ids, 7:13] = 0

    def _reset_task_transfer(self, env_ids):
        task_transfer_envs = (self.task_ids[env_ids] == self.cfg.task.TASK_TRANSFER)

        active_ids = env_ids[task_transfer_envs]
        inactive_ids = env_ids[~task_transfer_envs]

        if len(active_ids) > 0:
            pos = self.env_origins[active_ids].clone()

            # Reset front table
            self.front_table_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.front_table_offset[0]
            self.front_table_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.front_table_offset[1]
            self.front_table_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.front_table_offset[2]
            self.front_table_root_states[active_ids, 7:13] = 0

            # Reset back table
            self.back_table_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.back_table_offset[0]
            self.back_table_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.back_table_offset[1]
            self.back_table_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.back_table_offset[2]
            self.back_table_root_states[active_ids, 7:13] = 0

            # Reset box transfer
            self.box_transfer_root_states[active_ids, 0] = self.front_table_root_states[active_ids, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_transfer_range_x).to(self.device)
            self.box_transfer_root_states[active_ids, 1] = self.front_table_root_states[active_ids, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_transfer_range_y).to(self.device)
            self.box_transfer_root_states[active_ids, 2] = self.front_table_root_states[active_ids, 2] + 0.5 * self.cfg.asset.front_table_dims[2] + 0.5 * self.cfg.asset.box_transfer_size
            self.box_transfer_root_states[active_ids, 3] = 1
            self.box_transfer_root_states[active_ids, 4:] = 0

            # Reset goal
            self.box_transfer_goal_pos[active_ids, 0] = self.back_table_root_states[active_ids, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_transfer_range_x).to(self.device)
            self.box_transfer_goal_pos[active_ids, 1] = self.back_table_root_states[active_ids, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.box_transfer_range_y).to(self.device)
            self.box_transfer_goal_pos[active_ids, 2] = self.box_transfer_root_states[active_ids, 2]
        
        if len(inactive_ids) > 0:
            self.front_table_root_states[inactive_ids, 2] = self.hidden_z
            self.front_table_root_states[inactive_ids, 7:13] = 0

            self.back_table_root_states[inactive_ids, 2] = self.hidden_z
            self.back_table_root_states[inactive_ids, 7:13] = 0

            self.box_transfer_root_states[inactive_ids, 2] = self.hidden_z
            self.box_transfer_root_states[inactive_ids, 7:13] = 0

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # dynamic randomization
        # delay = torch.rand((self.num_envs, 1), device=self.device)
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
            arti_obj_force_buffer = torch.zeros((self.num_envs, self.arti_obj_num_dofs), device=self.device)
            full_force_buffer = torch.cat((self.torques, arti_obj_force_buffer), dim=1) # [num_envs, num_dofs + arti_obj_num_dofs]
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
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
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
        # self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.humanoid_root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        self.compute_visual_observations()
        
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        task_one_hot = F.one_hot(self.task_ids, num_classes=self.num_tasks).float()

        common_obs_buf = torch.cat((
            q,    # |A|
            dq,  # |A|
            self.actions,   # |A|
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            task_one_hot, # 8
        ), dim=-1)

        common_privileged_obs_buf = torch.cat((
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # |A|
            self.dof_vel * self.obs_scales.dof_vel,  # |A|
            self.actions,  # |A|
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            # stance_mask,  # 2
            contact_mask,  # 2
            task_one_hot, # 8
        ), dim=-1)

        task_specific_obs_buf = torch.zeros(self.num_envs, self.cfg.env.max_command_dim, device=self.device)
        task_specific_privileged_obs_buf = torch.zeros(self.num_envs, self.cfg.env.max_privileged_obs_dim, device=self.device)

        # Task ball observations
        task_ball_mask = (self.task_ids == self.cfg.task.TASK_BALL)
        if torch.any(task_ball_mask):
            ball_pos = self.ball_root_states[task_ball_mask, :3]
            torso_pos = self.rigid_state[task_ball_mask, self.torso_indices, :3].squeeze(1)
            ball_goal_diff = ball_pos - self.goal_pos[task_ball_mask]
            root_ball_diff = torso_pos - ball_pos

            goal_pos_obs = torch.flatten(self.goal_pos[task_ball_mask], start_dim=1) # [num_envs, 3]
            ball_pos_obs = torch.flatten(ball_pos, start_dim=1) # [num_envs, 3]
            torso_pos_obs = torch.flatten(torso_pos, start_dim=1) # [num_envs, 3]
            ball_goal_diff_obs = torch.flatten(ball_goal_diff, start_dim=1) # [num_envs, 3]
            root_ball_diff_obs = torch.flatten(root_ball_diff, start_dim=1) # [num_envs, 3]

            task_ball_obs = torch.cat((ball_goal_diff_obs, root_ball_diff_obs), dim=-1)
            task_specific_obs_buf[task_ball_mask, :6] = task_ball_obs

            task_ball_privileged_obs = torch.cat((
                goal_pos_obs, # 3
                ball_pos_obs, # 3
                torso_pos_obs, # 3
                ball_goal_diff_obs,  # 3
                root_ball_diff_obs,  # 3
            ), dim=-1)
            task_specific_privileged_obs_buf[task_ball_mask, :15] = task_ball_privileged_obs

        # Task box observations
        task_box_mask = (self.task_ids == self.cfg.task.TASK_BOX)
        if torch.any(task_box_mask):
            wrist_pos = self.rigid_state[task_box_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_root_states[task_box_mask, :3]
            diff = box_pos - self.box_goal_pos[task_box_mask]
            wrist_box_diff = wrist_pos - box_pos.unsqueeze(1) # [num_envs, 2, 3], two hands, position only
            
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 6]
            wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            box_goal_pos_obs = torch.flatten(self.box_goal_pos[task_box_mask], start_dim=1) # [num_envs, 3]
            box_pos_obs = torch.flatten(box_pos, start_dim=1) # [num_envs, 3]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 3]

            task_box_obs = torch.cat((diff_obs, wrist_box_diff_obs), dim=-1)
            task_specific_obs_buf[task_box_mask, :9] = task_box_obs

            task_box_privileged_obs = torch.cat((
                box_goal_pos_obs, # 3
                box_pos_obs, # 3
                diff_obs,  # 3
                wrist_pos_obs, # 6
                wrist_box_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_box_mask, :21] = task_box_privileged_obs

        # Task button observations
        task_button_mask = (self.task_ids == self.cfg.task.TASK_BUTTON)
        if torch.any(task_button_mask):
            wrist_pos = self.rigid_state[task_button_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:, 0, :3] # [num_envs, 3], left hand, position only
            button_goal_pos = self.button_goal_pos[task_button_mask, :3] # [num_envs, 3]
            wrist_button_diff = wrist_pos - button_goal_pos # [num_envs, 3], left hand, position only

            task_button_obs = wrist_button_diff
            task_specific_obs_buf[task_button_mask, :3] = task_button_obs

            task_button_privileged_obs = torch.cat((
                button_goal_pos, # 3
                wrist_pos, # 3
                wrist_button_diff, # 3
            ), dim=-1)
            task_specific_privileged_obs_buf[task_button_mask, :9] = task_button_privileged_obs

        # Task cabinet observations
        task_cabinet_mask = (self.task_ids == self.cfg.task.TASK_CABINET)
        if torch.any(task_cabinet_mask):
            wrist_pos = self.rigid_state[task_cabinet_mask][:, self.wrist_indices, :3] # [num_envs, 2, 3], two hands
            # torso_pos = self.rigid_state[task_cabinet_mask, self.torso_indices, :3].squeeze(1) # [num_envs, 3]
            arti_obj_pos = self.arti_obj_root_states[task_cabinet_mask, :3] # [num_envs, 3]
            arti_obj_dof_pos = self.arti_obj_dof_state[task_cabinet_mask][:, :, 0] # [num_envs, 2]
            arti_obj_dof_goal = self.arti_obj_dof_goal # 0
            arti_obj_dof_diff_obs = arti_obj_dof_pos - arti_obj_dof_goal # [num_envs, 2]
            wrist_arti_obj_diff = wrist_pos - arti_obj_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_arti_obj_diff_obs = torch.flatten(wrist_arti_obj_diff, start_dim=1) # [num_envs, 6]

            task_cabinet_obs = torch.cat((arti_obj_dof_diff_obs, wrist_arti_obj_diff_obs), dim=-1)
            task_specific_obs_buf[task_cabinet_mask, :8] = task_cabinet_obs

            task_cabinet_privileged_obs = torch.cat((
                arti_obj_dof_diff_obs, # 2
                wrist_arti_obj_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_cabinet_mask, :8] = task_cabinet_privileged_obs

        # Task carry observations
        task_carry_mask = (self.task_ids == self.cfg.task.TASK_CARRY)
        if torch.any(task_carry_mask):
            wrist_pos = self.rigid_state[task_carry_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_root_states[task_carry_mask, :3]
            diff = box_pos - self.box_goal_pos[task_carry_mask]
            wrist_box_diff = wrist_pos - box_pos.unsqueeze(1) # [num_envs, 2, 3], two hands, position only
            
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 6]
            wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            box_goal_pos_obs = torch.flatten(self.box_goal_pos[task_carry_mask], start_dim=1) # [num_envs, 3]
            box_pos_obs = torch.flatten(box_pos, start_dim=1) # [num_envs, 3]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 3]

            task_carry_obs = torch.cat((diff_obs, wrist_box_diff_obs), dim=-1)
            task_specific_obs_buf[task_carry_mask, :9] = task_carry_obs

            task_carry_privileged_obs = torch.cat((
                box_goal_pos_obs, # 3
                box_pos_obs, # 3
                diff_obs,  # 3
                wrist_pos_obs, # 6
                wrist_box_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_carry_mask, :21] = task_carry_privileged_obs


        # Task lift observations
        task_lift_mask = (self.task_ids == self.cfg.task.TASK_LIFT)
        if torch.any(task_lift_mask):
            wrist_pos = self.rigid_state[task_lift_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_root_states[task_lift_mask, :3]
            diff = box_pos - self.box_goal_pos[task_lift_mask]
            wrist_box_diff = wrist_pos - box_pos.unsqueeze(1) # [num_envs, 2, 3], two hands, position only
            
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 6]
            wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            box_goal_pos_obs = torch.flatten(self.box_goal_pos[task_lift_mask], start_dim=1) # [num_envs, 3]
            box_pos_obs = torch.flatten(box_pos, start_dim=1) # [num_envs, 3]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 3]

            task_lift_obs = torch.cat((diff_obs, wrist_box_diff_obs), dim=-1)
            task_specific_obs_buf[task_lift_mask, :9] = task_lift_obs

            task_lift_privileged_obs = torch.cat((
                box_goal_pos_obs, # 3
                box_pos_obs, # 3
                diff_obs,  # 3
                wrist_pos_obs, # 6
                wrist_box_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_lift_mask, :21] = task_lift_privileged_obs


        # Task reach observations
        task_reach_mask = (self.task_ids == self.cfg.task.TASK_REACH)
        if torch.any(task_reach_mask):
            wrist_pos = self.rigid_state[task_reach_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            diff = wrist_pos - self.ref_wrist_pos[task_reach_mask] # [num_envs, 2, 7], two hands

            ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos[task_reach_mask], start_dim=1) # [num_envs, 14]
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 14]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 14]

            task_reach_obs = diff_obs
            task_specific_obs_buf[task_reach_mask, :14] = task_reach_obs

            task_reach_privileged_obs = torch.cat((
                ref_wrist_pos_obs,  # 14
                wrist_pos_obs,  # 14
                diff_obs,  # 14
            ), dim=-1)
            task_specific_privileged_obs_buf[task_reach_mask, :42] = task_reach_privileged_obs

        # Task transfer observations
        task_transfer_mask = (self.task_ids == self.cfg.task.TASK_TRANSFER)
        if torch.any(task_transfer_mask):
            wrist_pos = self.rigid_state[task_transfer_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_root_states[task_transfer_mask, :3]
            diff = box_pos - self.box_goal_pos[task_transfer_mask]
            wrist_box_diff = wrist_pos - box_pos.unsqueeze(1) # [num_envs, 2, 3], two hands, position only
            
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 6]
            wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            box_goal_pos_obs = torch.flatten(self.box_goal_pos[task_transfer_mask], start_dim=1) # [num_envs, 3]
            box_pos_obs = torch.flatten(box_pos, start_dim=1) # [num_envs, 3]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 3]

            task_transfer_obs = torch.cat((diff_obs, wrist_box_diff_obs), dim=-1)
            task_specific_obs_buf[task_transfer_mask, :9] = task_transfer_obs

            task_transfer_privileged_obs = torch.cat((
                box_goal_pos_obs, # 3
                box_pos_obs, # 3
                diff_obs,  # 3
                wrist_pos_obs, # 6
                wrist_box_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_transfer_mask, :21] = task_transfer_privileged_obs

        # Concat
        obs_buf = torch.cat((task_specific_obs_buf, common_obs_buf), dim=-1)
        self.privileged_obs_buf = torch.cat((task_specific_privileged_obs_buf, common_privileged_obs_buf), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.humanoid_root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
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

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        
        # if the ball hits the goal, reset the env
        ball_pos = self.ball_root_states[:, :3]
        goal_pos = self.goal_pos
        ball_goal_diff = ball_pos - goal_pos # [envs, 3]
        ball_goal_dist = torch.norm(ball_goal_diff, dim=1)
        self.reset_buf |= ball_goal_dist < self.cfg.commands.ranges.threshold

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        new_tasks = torch.randint(0, self.num_tasks, (len(env_ids),), device=self.device)
        self.task_ids[env_ids] = new_tasks
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #
    
    # Task ball rewards
    def _reward_torso_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BALL)

        if torch.any(mask):
            torso_pos = self.rigid_state[mask][:, self.torso_indices, :3].squeeze(1) # [envs, 3]
            torso_ori_ball_pos_diff = self.ori_ball_pos[mask] - torso_pos
            torso_ori_ball_pos_diff = torso_ori_ball_pos_diff[:, :2] # only xy
            torso_ori_ball_pos_error = torch.mean(torch.abs(torso_ori_ball_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * torso_ori_ball_pos_error)
            error[mask] = torso_ori_ball_pos_error

        return reward, error
    
    def _reward_ball_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BALL)

        if torch.any(mask):
            ball_goal_diff = self.ball_root_states[mask, :3] - self.goal_pos[mask]
            ball_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)

            reward[mask] = torch.exp(-1 * ball_goal_error)
            error[mask] = ball_goal_error

        return reward, error

    # Task box rewards
    def _reward_wrist_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BOX)

        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error

    def _reward_box_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BOX)

        if torch.any(mask):
            box_pos_diff = self.box_root_states[mask, :3] - self.box_goal_pos[mask]
            box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * box_pos_error)
            error[mask] = box_pos_error

        return reward, error

    def _reward_wrist_box_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BOX)

        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_root_states[mask, :3] # [num_envs, 3]
            wrist_box_diff = wrist_pos - box_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_box_error)
            error[mask] = wrist_box_error

        return reward, error
    
    # Task button rewards
    def _reward_wrist_button_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BUTTON)

        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error

    def _reward_wrist_button_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BUTTON)

        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:, 0, :3] # [num_envs, 3], left hand, position only
            button_goal_pos = self.button_goal_pos[mask, :3] # [num_envs, 3]
            wrist_button_diff = wrist_pos - button_goal_pos # [num_envs, 3]
            wrist_button_error = torch.mean(torch.abs(wrist_button_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_button_error)
            error[mask] = wrist_button_error

        return reward, error

    def _reward_right_arm_default(self):
        """
        Calculates the reward for keeping right arm joint positions close to default positions.
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_BUTTON)

        if torch.any(mask):
            right_shoulder_pitch_index = 15
            joint_diff = self.dof_pos[mask] - self.default_joint_pd_target
            right_arm_diff = joint_diff[:, right_shoulder_pitch_index:] # start from right shoulder pitch
            right_arm_error = torch.mean(torch.abs(right_arm_diff), dim=1)

            reward[mask] = torch.exp(-4 * right_arm_error)
            error[mask] = right_arm_error

        return reward, error

    # Task cabinet rewards
    def _reward_torso_arti_obj_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_CABINET)

        if torch.any(mask):
            torso_pos = self.rigid_state[mask][:, self.torso_indices, :3].squeeze(1) # [num_envs, 3]
            arti_obj_pos = self.arti_obj_root_states[mask, :3] # [num_envs, 3]
            torso_arti_obj_diff = arti_obj_pos - torso_pos # [num_envs, 3]
            torso_arti_obj_distance = torch.norm(torso_arti_obj_diff, dim=1) # [num_envs]
            torso_arti_obj_distance[torso_arti_obj_distance < 0.1] = 0 # ignore small distance

            reward[mask] = torch.exp(-4 * torso_arti_obj_distance)
            error[mask] = torso_arti_obj_distance

        return reward, error

    def _reward_wrist_arti_obj_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_CABINET)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3] # [num_envs, 2, 3], two hands
            arti_obj_pos = self.arti_obj_root_states[mask, :3] # [num_envs, 3]
            wrist_arti_obj_diff = wrist_pos - arti_obj_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_arti_obj_diff = torch.flatten(wrist_arti_obj_diff, start_dim=1) # [num_envs, 6]
            wrist_arti_obj_error = torch.mean(torch.abs(wrist_arti_obj_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_arti_obj_error)
            error[mask] = wrist_arti_obj_error

        return reward, error

    def _reward_arti_obj_dof(self):
        """
        Calculates the reward based on the difference between the current arti_obj dof positions and the target dof positions.
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_CABINET)
    
        if torch.any(mask):
            arti_obj_dof_diff = self.arti_obj_dof_state[mask][:, :, 0] - self.arti_obj_dof_goal # [num_envs, 2]
            arti_obj_dof_error = torch.mean(torch.abs(arti_obj_dof_diff), dim=1)

            reward[mask] = torch.exp(-4 * arti_obj_dof_error)
            error[mask] = arti_obj_dof_error

        return reward, error

    # Task carry rewards
    def _reward_wrist_carry_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_CARRY)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error
    
    def _reward_box_carry_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_CARRY)
    
        if torch.any(mask):
            box_pos_diff = self.box_root_states[mask, :3] - self.box_goal_pos[mask]
            box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * box_pos_error)
            error[mask] = box_pos_error

        return reward, error

    def _reward_wrist_box_carry_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_CARRY)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_carry_root_states[mask, :3] # [num_envs, 3]
            box_handle_left = box_pos.clone()
            box_handle_right = box_pos.clone()
            # box_handle_left[:, 1] += 0.4 * self.cfg.asset.box_size[1]
            # box_handle_left[:, 2] += 0.25 * self.cfg.asset.box_size[2]
            # box_handle_right[:, 1] -= 0.4 * self.cfg.asset.box_size[1]
            # box_handle_right[:, 2] += 0.25 * self.cfg.asset.box_size[2]
            box_handle_pos = torch.stack([box_handle_left, box_handle_right], dim=1) # [num_envs, 2, 3]
            wrist_box_diff = wrist_pos - box_handle_pos # [num_envs, 2, 3]
            wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_box_error)
            error[mask] = wrist_box_error

        return reward, error
    
    # Task lift rewards
    def _reward_wrist_lift_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_LIFT)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error

    def _reward_box_lift_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_LIFT)
    
        if torch.any(mask):
            box_pos_diff = self.box_lift_root_states[mask, :3] - self.box_goal_pos[mask]
            box_pos_diff = box_pos_diff[:, 2:3] # only z axis
            box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * box_pos_error)
            error[mask] = box_pos_error

        return reward, error

    def _reward_wrist_box_lift_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_LIFT)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_lift_root_states[mask, :3] # [num_envs, 3]
            box_handle_left = box_pos.clone()
            box_handle_right = box_pos.clone()
            # box_handle_left[:, 1] += 0.4 * self.cfg.asset.box_size[1]
            # box_handle_left[:, 2] += 0.25 * self.cfg.asset.box_size[2]
            # box_handle_right[:, 1] -= 0.4 * self.cfg.asset.box_size[1]
            # box_handle_right[:, 2] += 0.25 * self.cfg.asset.box_size[2]
            box_handle_pos = torch.stack([box_handle_left, box_handle_right], dim=1) # [num_envs, 2, 3]
            wrist_box_diff = wrist_pos - box_handle_pos # [num_envs, 2, 3]
            wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_box_error)
            error[mask] = wrist_box_error

        return reward, error

    # Task reach rewards
    def _reward_wrist_reach_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_REACH)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error

    # Task transfer rewards
    def _reward_wrist_transfer_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_TRANSFER)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error
    
    def _reward_box_transfer_pos(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_TRANSFER)
    
        if torch.any(mask):
            box_pos_diff = self.box_root_states[mask][:, :3] - self.box_goal_pos[mask]
            box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * box_pos_error)
            error[mask] = box_pos_error

        return reward, error
    
    def _reward_wrist_box_transfer_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.task.TASK_TRANSFER)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            box_pos = self.box_root_states[mask, :3] # [num_envs, 3]
            wrist_box_diff = wrist_pos - box_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_box_error)
            error[mask] = wrist_box_error

        return reward, error

    # Common rewards
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        # Penalize distance to target joint angles
        dof_pos_diff = self.dof_pos - self.ref_dof_pos
        # dof_pos_diff = dof_pos_diff[:, 10:]
        target_jt_error = torch.mean(torch.abs(dof_pos_diff), dim=1)
        return torch.exp(-4 * target_jt_error), target_jt_error

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
    
    # @jhcao
    def _reward_elbow_distance(self):
        """
        Calculates the reward based on the distance between the elbow of the humanoid.
        """
        elbow_pos = self.rigid_state[:, self.elbow_indices, :2]
        elbow_dist = torch.norm(elbow_pos[:, 0, :] - elbow_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(elbow_dist - fd, -0.5, 0.)
        d_max = torch.clamp(elbow_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, elbow_dist
    
    # # @jhcao
    # def _reward_elbow_torso_distance(self):
    #     """
    #     Calculates the reward based on the distance of both elbow to the torso.
    #     """
    #     elbow_pos = self.rigid_state[:, self.elbow_indices, :2]
    #     torso_pos = self.rigid_state[:, self.torso_indices, :2]
    #     elbow_torso_dist_l = torch.norm(elbow_pos[:,0,:] - torso_pos[:,0,:], dim=1)
    #     elbow_torso_dist_r = torch.norm(elbow_pos[:,1,:] - torso_pos[:,0,:], dim=1)
    #     fd = 0.00
    #     max_fd = 0.10
    #     d_min_l = torch.clamp(elbow_torso_dist_l - fd, -0.5, 0.)
    #     d_max_l = torch.clamp(elbow_torso_dist_l - max_fd, 0, 0.5)
    #     d_min_r = torch.clamp(elbow_torso_dist_r - fd, -0.5, 0.)
    #     d_max_r = torch.clamp(elbow_torso_dist_r - max_fd, 0, 0.5)
    #     return (torch.exp(-torch.abs(d_min_l) * 100) + torch.exp(-torch.abs(d_max_l) * 100)) / 2 + (torch.exp(-torch.abs(d_min_r) * 100) + torch.exp(-torch.abs(d_max_r) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 5:7]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    
    def _reward_upper_body_pos(self):
        """
        Calculates the reward for keeping upper body joint positions close to default positions.
        """
        torso_index = 10
        joint_diff = self.dof_pos - self.default_joint_pd_target
        upper_body_diff = joint_diff[:, torso_index:torso_index+1] # start from torso, but only torso
        upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
        return torch.exp(-4 * upper_body_error), upper_body_error
    
    # def _reward_upper_body_pos(self):
    #     """
    #     Calculates the reward for keeping upper body joint positions close to default positions.
    #     """
    #     joint_diff = self.dof_pos - self.default_joint_pd_target
    #     upper_body_diff = joint_diff[:, 10:] # start from torso
    #     upper_body_diff = torch.norm(upper_body_diff, dim=1)
    #     upper_body_diff = torch.clamp(upper_body_diff - 0.1, 0, 50)
    #     return torch.exp(-upper_body_diff * 100)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.humanoid_root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.humanoid_root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_diff = self.commands[:, :2] - self.base_lin_vel[:, :2]
        lin_vel_error = torch.sum(torch.square(
            lin_vel_diff), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma), torch.mean(torch.abs(lin_vel_diff), dim=1)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        ang_vel_diff = self.commands[:, 2] - self.base_ang_vel[:, 2]
        ang_vel_error = torch.square(
            ang_vel_diff)
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma), torch.abs(ang_vel_diff)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
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