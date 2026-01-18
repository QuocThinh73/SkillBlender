from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1UnifiedTaskCfg(LeggedRobotCfg):
    class task():
        num_tasks = 8

        TASK_BALL = 0
        TASK_BOX = 1
        TASK_BUTTON = 2
        TASK_CABINET = 3
        TASK_CARRY = 4
        TASK_LIFT = 5
        TASK_REACH = 6
        TASK_TRANSFER = 7

    class human(LeggedRobotCfg.human):
        freq = 1

    class env(LeggedRobotCfg.env):
        # change the observation dim
        num_actions = 19
        frame_stack = 1
        c_frame_stack = 3
        command_dim = 8
        num_single_obs = 3 * num_actions + 6 + command_dim # see `obs_buf = torch.cat(...)` for details
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 3 * num_actions + 18 + 8
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        
        num_envs = 16
        episode_length_s = 8  # episode length in seconds
        use_ref_actions = False
        env_spacing = 10.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf'

        hidden_z = -10.0

        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        elbow_name = "elbow"
        torso_name = "torso"
        wrist_name = "wrist"

        terminate_after_contacts_on = ['pelvis', 'torso', 'shoulder', 'elbow', 'hip']
        penalize_contacts_on = ["hip", 'knee', 'pelvis', 'torso', 'shoulder', 'elbow']
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        fix_base_link = False
        collapse_fixed_joints = False
        
        # Task ball
        ## Door
        door_dims = [
            [0.05, 4.0, 2.0],
            [1.0, 0.05, 2.0],
            [1.0, 0.05, 2.0]
        ]
        door_offsets = [
            [5.0, 0.0, 1.0],
            [4.5, 2.0, 1.0],
            [4.5, -2.0, 1.0]
        ]
        ## Ball
        ball_size = 0.2
        ball_range_x = [0.5, 1.0]
        ball_range_y = [-0.3, 0.3]
        ball_range_mass = [0.3, 0.5]

        # Task box
        ## Table
        table_dims = [0.9, 1.5, 0.05]
        table_offset = [1.0, 0.0, 0.975]
        ## Box
        box_size = 0.1
        box_range_x = [-0.45, -0.35]
        box_range_y = [-0.3, 0.3]

        # Task button
        ## Wall
        wall_dims = [0.05, 2.0, 3.0]
        wall_offset = [1.5, 0.0, 1.5]
        ## Button
        button_ori_z = 1.0

        # Task cabinet
        ## Cabinet
        gapartnet_root = "resources/objects/gapartnet/"
        gapartnet_id = 45159
        arti_obj_offset = [1.5, 0.0, 1.0]
        arti_obj_dof_default = 1.0
        arti_obj_scale = 0.5

        # Task carry
        ## Box carry
        box_carry_size = [0.5, 0.1, 1.0]
        box_carry_offset_xy = [1.0, 0.0]
        box_carry_range_x = [-0.5, -0.3]
        box_carry_range_y = [-0.05, 0.05]
        box_carry_range_mass = [0.1, 2.0]

        # Task lift
        ## Box lift
        box_lift_size = [0.5, 0.1, 1.0]
        box_lift_offset_xy = [1.0, 0.0]
        box_lift_range_x = [-0.5, -0.3]
        box_lift_range_y = [-0.05, 0.05]
        box_lift_range_mass = [0.1, 2.0]

        # Task transfer
        ## Table
        front_table_dims = [0.9, 1.5, 0.05]
        front_table_offset = [1.0, 0.0, 0.975]
        back_table_dims = [0.9, 1.5, 0.05]
        back_table_offset = [-1.0, 0.0, 0.975]
        ## Box transfer
        box_transfer_size = 0.1
        box_transfer_mass = 0.01
        box_transfer_range_x = [-0.45, -0.35]
        box_transfer_range_y = [-0.3, 0.3]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        # mesh_type = 'trimesh'
        # curriculum = True
        
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
        pos = [0.0, 0.0, 1.0] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
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
        curriculum = False # if true: curriculum update of commands

        class ranges:
            lin_vel_x = [-0, 0]  # min max [m/s]
            lin_vel_y = [-0, 0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
            # Task ball
            ## goal, related to asset size
            goal_x = [5.0, 5.0]
            goal_y = [-2.0, 2.0]
            goal_z = [0, 0.5]
            threshold = 0.5
            # Task box
            ## wrist pos command ranges
            wrist_max_radius = 0.25
            l_wrist_pos_x = [-0.10, 0.25]
            l_wrist_pos_y = [-0.10, 0.25]
            l_wrist_pos_z = [-0.25, 0.25]
            r_wrist_pos_x = [-0.10, 0.25]
            r_wrist_pos_y = [-0.25, 0.10]
            r_wrist_pos_z = [-0.25, 0.25]
            ## box goal pos command ranges
            box_pos_x = [0, 0.1]
            box_pos_y = [-0.5, 0.5]
            # Task button
            ## wrist pos command ranges
            wrist_max_radius = 0.25
            l_wrist_pos_x = [-0.10, 0.25]
            l_wrist_pos_y = [-0.10, 0.25]
            l_wrist_pos_z = [-0.25, 0.25]
            r_wrist_pos_x = [-0.10, 0.25]
            r_wrist_pos_y = [-0.25, 0.10]
            r_wrist_pos_z = [-0.25, 0.25]
            ## button pos command ranges
            button_pos_y = [-0.5, 0.5]
            button_pos_z = [-0.5, 0.5]
            # Task cabinet
            wrist_max_radius = 0.25
            l_wrist_pos_x = [-0.10, 0.25]
            l_wrist_pos_y = [-0.10, 0.25]
            l_wrist_pos_z = [-0.25, 0.25]
            r_wrist_pos_x = [-0.10, 0.25]
            r_wrist_pos_y = [-0.25, 0.10]
            r_wrist_pos_z = [-0.25, 0.25]
            # Task carry
            box_carry_pos_x = [0.3, 1.0]
            box_carry_pos_y = [-0.3, 0.3]
            box_carry_pos_z = [0.3, 0.6] # maybe too large?
            # Task lift
            # box goal pos command ranges
            box_lift_pos_z = [0.3, 0.6] # maybe too large?
            # Task reach
            # wrist pos command ranges
            wrist_max_radius = 0.25
            l_wrist_pos_x = [-0.10, 0.25]
            l_wrist_pos_y = [-0.10, 0.25]
            l_wrist_pos_z = [-0.25, 0.25]
            r_wrist_pos_x = [-0.10, 0.25]
            r_wrist_pos_y = [-0.25, 0.10]
            r_wrist_pos_z = [-0.25, 0.25]
            # center
            max_center_distance = 2
            center_offset_x = [-2, 2]
            center_offset_y = [-2, 2]
            center_offset_z = [-0.5, 0.5]
            # Task transfer
            # wrist pos command ranges
            wrist_max_radius = 0.25
            l_wrist_pos_x = [-0.10, 0.25]
            l_wrist_pos_y = [-0.10, 0.25]
            l_wrist_pos_z = [-0.25, 0.25]
            r_wrist_pos_x = [-0.10, 0.25]
            r_wrist_pos_y = [-0.25, 0.10]
            r_wrist_pos_z = [-0.25, 0.25]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.89
        min_dist = 0.2
        max_dist = 0.5
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
            # TODO: 1. stand_still 2. joint_pos*2 3. add command input
            # reference motion tracking
            # joint_pos = 5
            # Task ball
            torso_pos = 1
            ball_pos = 5
            # Task box
            box_pos = 5
            wrist_box_distance = 5
            # Task button
            wrist_button_distance = 5
            right_arm_default = 0.5
            # Task cabinet
            wrist_arti_obj_distance = 5
            arti_obj_dof = 5
            # Task carry
            box_carry_pos = 5
            wrist_box_carry_distance = 5
            # Task lift
            box_lift_pos = 5
            wrist_box_lift_distance = 5
            # Task reach
            wrist_pos = 5
            # Task transfer
            box_transfer_pos = 5
            wrist_box_transfer_distance = 1
            # feet_clearance = 0
            # feet_contact_number = 0
            # # gait
            # feet_air_time = 0
            # foot_slip = -0.05
            # feet_distance = 0.5
            # knee_distance = 0.2
            # # elbow_distance = 0.4
            # # elbow_torso_distance = 0.4
            # # contact
            # feet_contact_forces = -0.01
            # # vel tracking
            # tracking_lin_vel = 0.
            # tracking_ang_vel = 0.
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5 * 2
            # # base pos
            # default_joint_pos = 0.5
            # upper_body_pos = 0.5
            # orientation = 1.
            # base_height = 0.2
            # base_acc = 0.2
            # energy
            # action_smoothness = -0.002
            # torques = -1e-5
            # dof_vel = -5e-4
            # dof_acc = -1e-7
            # collision = -0.2
            #### humanplus ####
            # lin_vel_z = -0.1
            # ang_vel_xy = -0.1

    class sensor(LeggedRobotCfg.sensor):
        enable_sensor = False
        class camera(LeggedRobotCfg.sensor.camera):
            height = 48
            width = 64
            offset = (0.1, 0.0, 0.9) # relative to root
            axis = (0, 1, 0) # camera axis
            angle = 45 # camera angle

class H1UnifiedTaskCfgPPO(LeggedRobotCfgPPO):
    pass