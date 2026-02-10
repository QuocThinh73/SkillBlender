from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1MultitaskCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # Task
        num_tasks = 5
        TASK_REACH = 0
        TASK_BUTTON = 1
        TASK_CABINET = 2
        TASK_BOX = 3
        TASK_BALL = 4

        # Observation
        num_actions = 19
        frame_stack = 1
        c_frame_stack = 3
        num_proprioception_obs = 3 * num_actions + 9
        num_task_obs = 32
        num_phase_obs = num_tasks
        num_single_obs = num_proprioception_obs + num_task_obs + num_phase_obs # see `obs_buf = torch.cat(...)` for details
        num_observations = int(frame_stack * num_single_obs)
        num_proprioception_privileged_obs = 3 * num_actions + 18
        num_task_privileged_obs = 62
        num_phase_privileged_obs = num_tasks
        num_single_privileged_obs = num_proprioception_privileged_obs + num_task_privileged_obs + num_phase_privileged_obs
        num_privileged_obs = int(c_frame_stack * num_single_privileged_obs)

        command_dim = num_task_obs
        
        num_envs = 2048
        env_spacing = 10.0

        # Episode length
        reach_length_s = 8.0
        button_length_s = 8.0
        cabinet_length_s = 8.0
        box_length_s = 12.0
        ball_length_s = 12.0
        episode_length_s = reach_length_s + button_length_s + cabinet_length_s + box_length_s + ball_length_s

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf"

        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        elbow_name = "elbow"
        torso_name = "torso"
        wrist_name = "wrist"

        terminate_after_contacts_on = [
            'pelvis',
            'torso',
            'waist',
            'shoulder',
            'elbow',
            'knee',
        ]
        penalize_contacts_on = [
            'hip', 
            'knee', 
            'pelvis', 
            'torso', 
            'shoulder', 
            'elbow'
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        fix_base_link = False
        collapse_fixed_joints = False

        # Task ball
        ## Goal assets
        goal_dims = [
            [0.05, 4.0, 2.0],
            [1.0, 0.05, 2.0],
            [1.0, 0.05, 2.0]
        ]
        goal_offsets = [
            [5.0, 0.0, 1.0],
            [4.5, 2.0, 1.0],
            [4.5, -2.0, 1.0]
        ]
        ## Ball assets
        ball_size = 0.2
        ball_range_x = [2.0, 2.5]
        ball_range_y = [-0.3, 0.3]
        ball_range_mass = [0.3, 0.5]
        # Task button
        ## Wall assets
        wall_dims = [0.05, 2.0, 3.0]
        wall_offsets = [-1.5, 0.0, 1.5]
        ## Button assets
        button_ori_z = 1.0
        # Task box
        ## Table assets
        table_dims = [1.5, 0.9, 0.05]
        table_offsets = [0.0, 2.0, 0.975]
        ## Small box assets
        small_box_size = 0.1
        small_box_range_x = [-0.3, 0.3]
        small_box_range_y = [-0.45, -0.35]
        # Task cabinet
        ## Cabinet
        gapartnet_root = "resources/objects/gapartnet/"
        gapartnet_id = 45159
        cabinet_offsets = [0, -2.0, 1.0]
        cabinet_dof_default = 1.0
        cabinet_scale = 0.5

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        
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
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 10
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        curriculum = False # if true: curriculum update of commands

        class ranges:
            lin_vel_x = [-0, 0]     # min max [m/s]
            lin_vel_y = [-0, 0]     # min max [m/s]
            ang_vel_yaw = [-0, 0]   # min max [rad/s]
            heading = [-0, 0]
            # Task ball
            ball_goal_x = [5.0, 5.0]
            ball_goal_y = [-2.0, 2.0]
            ball_goal_z = [0, 0.5]
            threshold = 0.5
            # Task box
            small_box_x = [-0.5, 0.5]
            small_box_y = [0, 0.1]
            # Task button
            button_goal_y = [-0.5, 0.5]
            button_goal_z = [-0.5, 0.5]
            # Task reach
            center_goal_x = [-1, 1]
            center_goal_y = [-1, 1]
            center_goal_z = [0.75, 1.25]
            offset_x = [-0.15, 0.15]
            offset_y = [-0.15, 0.15]
            offset_z = [-0.1, 0.1]

    class rewards:
        min_dist = 0.05
        max_dist = 0.25
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.95
        base_height_target = 0.89

        cycle_time = 0.64

        only_positive_rewards = False

        class scales:
            # Task rewards
            ## Task reach
            ### Main reward
            reach_total = 5.0
            ## Task button
            ### Main reward
            button_total = 5.0
            ## Task cabinet
            ### Main goal
            cabinet_total = 5.0
            ## Task box
            ### Main goal
            box_total = 5.0
            ## Task ball
            ### Main goal
            ball_total = 5.0

            # Base rewards
            orientation = 1.0
            base_height = 0.2
            feet_distance = 0.5
            knee_distance = 0.2
            collision = -0.2
            action_smoothness = -0.002

            # HumanPlus
            lin_vel_z = -0.05
            ang_vel_xy = -0.05
            action_rate = -0.05
            termination = -100.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            torque_limits = -0.1
            stumble = -0.1
            # stand_still = 0.0
            # target_jt = 0.0

        class task_thresholds:
            reach_near = 0.45
            reach_far  = 1.20

            button_near = 0.40
            button_far  = 1.00

            cabinet_near = 0.60
            cabinet_far  = 1.50

            box_near = 0.60
            box_far  = 1.50

            ball_near = 0.70
            ball_far  = 2.00

            cabinet_hand_near = 0.15
            cabinet_hand_far  = 0.40

            ball_push_near = 0.40
            ball_push_far  = 1.00

    class sensor(LeggedRobotCfg.sensor):
        enable_sensor = False
        class camera(LeggedRobotCfg.sensor.camera):
            height = 48
            width = 64
            offset = (0.1, 0.0, 0.9) # relative to root
            axis = (0, 1, 0) # camera axis
            angle = 45 # camera angle

class H1MultitaskCfgPPO(LeggedRobotCfgPPO):
    pass

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1MultitaskCfg(LeggedRobotCfg):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        # HRL
        num_dofs = H1MultitaskCfg.env.num_actions
        frame_stack = H1MultitaskCfg.env.frame_stack
        command_dim = H1MultitaskCfg.env.command_dim
        # Expert skills
        skill_dict = {
            'h1_walking': {
                "experiment_name": "h1_walking",
                "load_run": "0000_best",
                "checkpoint": -1,
                "low_high": (-2, 2)
            },
            'h1_reaching': {
                "experiment_name": "h1_reaching",
                "load_run": "0000_best",
                "checkpoint": -1,
                "low_high": (-1, 1)
            },
            'h1_squatting': {
                "experiment_name": "h1_squatting",
                "load_run": "0000_best",
                "checkpoint": -1,
                "low_high": (-1, 1)
            },
            'h1_stepping': {
                "experiment_name": "h1_stepping",
                "load_run": "0000_best",
                "checkpoint": -1,
                "low_high": (-1, 1)
            }
        }

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCriticHierarchical'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 30001 # 3001  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'h1_task_box'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and ckpt
