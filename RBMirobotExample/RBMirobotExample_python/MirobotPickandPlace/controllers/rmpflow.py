import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:
        # TODO: chamge the follow paths

        # # laptop
        # self._desc_path = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/MirobotFollowTarget/"
        # self._urdf_path = "/home/kimsooyoung/Downloads/Source/mirobot_ros2/mirobot_description/urdf/"

        # # desktop
        # self._desc_path = "/home/kimsooyoung/Downloads/source/RoadBalanceEdu/rb_issac_tutorial/RoadBalanceEdu/MirobotPickandPlace/"
        # self._urdf_path = "/home/kimsooyoung/Downloads/source/mirobot_ros2/mirobot_description/urdf/"

        # demo desktop
        self._desc_path = "/home/kimsooyoung/Downloads/Source/RB_MirobotExample/RBMirobotExample/RBMirobotExample_python/MirobotFollowTarget/"
        self._urdf_path = "/home/kimsooyoung/Downloads/Source/mirobot_ros2/mirobot_description/urdf/"


        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=self._desc_path+"rmpflow/robot_descriptor.yaml",
            rmpflow_config_path=self._desc_path+"rmpflow/mirrobot_rmpflow_common.yaml",
            urdf_path=self._urdf_path+"mirobot_urdf_2.urdf",
            end_effector_frame_name="Link6",
            maximum_substep_size=0.00334
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )