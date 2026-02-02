from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1MultitaskCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # Observation
        num_actions = 19
        
        num_envs = 4
        env_spacing = 10.0

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf"

        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        elbow_name = "elbow"
        torso_name = "torso"
        wrist_name = "wrist"

        terminate_after_contacts_on = []

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
        # Task button
        ## Wall assets
        wall_dims = [0.05, 2.0, 3.0]
        wall_offsets = [-1.5, 0.0, 1.5]
        # Task box
        ## Table assets
        table_dims = [1.5, 0.9, 0.05]
        table_offsets = [0.0, 2.0, 0.975]
        ## Small box assets
        small_box_size = 0.1
        # Task cabinet
        ## Cabinet
        gapartnet_root = "resources/objects/gapartnet/"
        gapartnet_id = 45159
        cabinet_offsets = [0, -2.0, 1.0]
        cabinet_dof_default = 1.0
        cabinet_scale = 0.5

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

        measure_heights = False
        
    class init_state(LeggedRobotCfg.init_state):
        pass

    class control(LeggedRobotCfg.control):
        pass

    class sim(LeggedRobotCfg.sim):
        pass

    class commands(LeggedRobotCfg.commands):
        

        class ranges:
            lin_vel_x = [-0, 0]     # min max [m/s]
            lin_vel_y = [-0, 0]     # min max [m/s]
            ang_vel_yaw = [-0, 0]   # min max [rad/s]
            heading = [-0, 0]

    class rewards:
        only_positive_rewards = True

        class scales:
            pass

    class sensor(LeggedRobotCfg.sensor):
        pass

class H1MultitaskCfgPPO(LeggedRobotCfgPPO):
    pass