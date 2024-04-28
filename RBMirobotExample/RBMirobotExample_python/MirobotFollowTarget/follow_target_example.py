# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators import SingleManipulator

import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np
import carb

from .ik_solver import KinematicsSolver
from .controllers.rmpflow import RMPFlowController

# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "mirobot_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )

        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        self._robot_path = self._server_root + "/Projects/RBROS2/mirobot_ros2/mirobot_description/urdf/mirobot_urdf_2/mirobot_urdf_2_ee.usd"
        self._joints_default_positions = np.zeros(6)

        return

    def set_robot(self) -> SingleManipulator:

        # add robot to the scene
        add_reference_to_stage(usd_path=self._robot_path, prim_path="/World/mirobot")

        # define the gripper
        self._gripper = SurfaceGripper(
            end_effector_prim_path="/World/mirobot/ee_link",
            translate=0.02947,
            direction="x",
            # kp=1.0e4,
            # kd=1.0e3,
            # disable_gravity=False,
        )
        self._gripper.set_force_limit(value=1.0e2)
        self._gripper.set_torque_limit(value=1.0e3)

        # define the manipulator
        manipulator = SingleManipulator(
            prim_path="/World/mirobot",
            name="mirobot",
            end_effector_prim_name="ee_link",
            gripper=self._gripper,
        )

        manipulator.set_joints_default_state(positions=self._joints_default_positions)

        return manipulator


class FollowTargetExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
    
        self._articulation_controller = None

        # simulation step counter
        self._sim_step = 0

        return

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        # We add the task to the world here
        my_task = FollowTarget(
            name="mirobot_follow_target", 
            target_position=np.array([0.15, 0, 0.15]),
            target_orientation=np.array([1, 0, 0, 0]),
        )
        self._world.add_task(my_task)

        return

    async def setup_post_load(self):
        self._world = self.get_world()

        self._task_params = self._world.get_task("mirobot_follow_target").get_params()
        self._target_name = self._task_params["target_name"]["value"]
        self._my_mirobot = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        
        # IK controller
        self._my_controller = KinematicsSolver(self._my_mirobot)
        
        # RMPFlow controller
        # self._my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=self._my_mirobot)

        self._articulation_controller = self._my_mirobot.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.sim_step_cb)
        return

    async def setup_post_reset(self):
        self._my_controller.reset()
        await self._world.play_async()
        return

    def sim_step_cb(self, step_size):
        
        observations = self._world.get_observations()

        pos = observations[self._target_name]["position"]
        ori = observations[self._target_name]["orientation"]

        # IK controller
        actions, succ = self._my_controller.compute_inverse_kinematics(
            target_position=pos
        )
        if succ:
            self._articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")

        # # RMPFlow controller
        # actions = self._my_controller.forward(
        #     target_end_effector_position=pos,
        #     target_end_effector_orientation=ori,
        # )
        # self._articulation_controller.apply_action(actions)

        return