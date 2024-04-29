from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.articulations import Articulation
from typing import Optional
import carb


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        #TODO: change the config path
        
        # desktop
        # my_path = "/home/kimsooyoung/Documents/IsaacSim/rb_issac_tutorial/RoadBalanceEdu/ManipFollowTarget/"
        # self._urdf_path = "/home/kimsooyoung/Downloads/USD/cobotta_pro_900/"
        
        # # lactop 
        # self._desc_path = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/MirobotFollowTarget/"
        # self._urdf_path = "/home/kimsooyoung/Downloads/Source/mirobot_ros2/mirobot_description/urdf/"

        # demo desktop
        self._desc_path = "/home/kimsooyoung/Downloads/Source/RB_MirobotExample/RBMirobotExample/RBMirobotExample_python/MirobotFollowTarget/"
        self._urdf_path = "/home/kimsooyoung/Downloads/Source/mirobot_ros2/mirobot_description/urdf/"

        self._kinematics = LulaKinematicsSolver(
            robot_description_path=self._desc_path+"rmpflow/robot_descriptor.yaml",
            urdf_path=self._urdf_path+"mirobot_urdf_2.urdf"
        )
        
        if end_effector_frame_name is None:
            end_effector_frame_name = "Link6"
        
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        
        return