# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1WalkingCfg(LeggedRobotCfg):
    """
    Configuration class for the G1 humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        num_actions = 21
        frame_stack = 1
        c_frame_stack = 3
        command_dim = 3
        num_single_obs = 3 * num_actions + 6 + command_dim # see `obs_buf = torch.cat(...)` for details
        num_observations = int(frame_stack * num_single_obs)
        num_single_privileged_obs = 4 * num_actions + 25
        num_privileged_obs = int(c_frame_stack * num_single_privileged_obs)
        
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_lock_waist_rev_1_0_modified.urdf'

        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        elbow_name = "elbow"
        torso_name = "torso_link"
        wrist_name = "rubber_hand"

        terminate_after_contacts_on = ['pelvis', 'torso', 'waist', 'shoulder', 'elbow', 'wrist']
        penalize_contacts_on = ["hip", 'knee']
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        fix_base_link = False
        collapse_fixed_joints = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        # curriculum = False
        mesh_type = 'trimesh'
        curriculum = True
        
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint' : -0.4,   
            'left_hip_roll_joint' : 0,   
            'left_hip_yaw_joint' : 0. ,                     
            'left_knee_joint' : 0.8,       
            'left_ankle_pitch_joint' : -0.4,     
            'left_ankle_roll_joint' : 0,   
            
            'right_hip_pitch_joint' : -0.4,    
            'right_hip_roll_joint' : 0,   
            'right_hip_yaw_joint' : 0.,                                    
            'right_knee_joint' : 0.8,                                             
            'right_ankle_pitch_joint': -0.4,                              
            'right_ankle_roll_joint' : 0,      
             
            'waist_yaw_joint' : 0., # index=12
            
            "left_shoulder_pitch_joint": 0.,
            "left_shoulder_roll_joint": 0.,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 0.,
            
            "right_shoulder_pitch_joint": 0.,
            "right_shoulder_roll_joint": 0.,
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 200,
                     'knee': 200,
                     'ankle': 20,
                     'waist': 200,
                     'shoulder': 40,
                     'elbow': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 5,
                     'ankle': 4,
                     'waist': 5,
                     'shoulder': 10,
                     'elbow': 10,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        curriculum = True # if true: curriculum update of commands

        class ranges:
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.728
        min_dist = 0.05
        max_dist = 0.25
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06       # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1. * 2
            feet_contact_number = 1.2 * 2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # elbow_distance = 0.4
            # elbow_torso_distance = 0.4
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2 * 2
            tracking_ang_vel = 1.1 * 2
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5 * 2
            # stand_still = -0.5
            # base pos
            default_joint_pos = 0.5
            upper_body_pos = 0.5 * 4
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.


class G1WalkingCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001 # 3001  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'g1_walking'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and ckpt