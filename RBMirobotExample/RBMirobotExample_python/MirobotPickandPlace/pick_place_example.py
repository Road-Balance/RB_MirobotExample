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
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.manipulators import SingleManipulator

from omni.isaac.dynamic_control import _dynamic_control as dc
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
import numpy as np
import omni
import carb

from .controllers.pick_place import PickPlaceController

def createRigidBody(stage, bodyType, boxActorPath, mass, scale, position, rotation, color):
    p = Gf.Vec3f(position[0], position[1], position[2])
    orientation = Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
    scale = Gf.Vec3f(scale[0], scale[1], scale[2])

    bodyGeom = bodyType.Define(stage, boxActorPath)
    bodyPrim = stage.GetPrimAtPath(boxActorPath)
    bodyGeom.AddTranslateOp().Set(p)
    bodyGeom.AddOrientOp().Set(orientation)
    bodyGeom.AddScaleOp().Set(scale)
    bodyGeom.CreateDisplayColorAttr().Set([color])

    UsdPhysics.CollisionAPI.Apply(bodyPrim)
    if mass > 0:
        massAPI = UsdPhysics.MassAPI.Apply(bodyPrim)
        massAPI.CreateMassAttr(mass)
    UsdPhysics.RigidBodyAPI.Apply(bodyPrim)
    UsdPhysics.CollisionAPI(bodyPrim)
    return bodyGeom


class PickandPlaceExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
    
        self._gripper = None
        self._my_mirobot = None
        self._articulation_controller = None

        # simulation step counter
        self._sim_step = 0
        self._target_position = np.array([0.2, -0.08, 0.06])

        return
    
    def setup_robot(self):

        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        self._robot_path = self._server_root + "/Projects/RBROS2/mirobot_ros2/mirobot_description/urdf/mirobot_urdf_2/mirobot_urdf_2_ee.usd"
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
        self._my_mirobot = self._world.scene.add(
            SingleManipulator(
                prim_path="/World/mirobot",
                name="mirobot",
                end_effector_prim_name="ee_link",
                gripper=self._gripper
            )
        )

        self._joints_default_positions = np.zeros(6)
        self._my_mirobot.set_joints_default_state(positions=self._joints_default_positions)

    def setup_bin(self):
        self._nucleus_server = get_assets_root_path()
        table_path = self._nucleus_server + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        add_reference_to_stage(usd_path=table_path, prim_path=f"/World/bin")
        self._bin_initial_position = np.array([0.2, 0.08, 0.06]) / get_stage_units()
        self._packing_bin = self._world.scene.add(
            GeometryPrim(
                prim_path="/World/bin", 
                name=f"packing_bin", 
                position=self._bin_initial_position,
                orientation=euler_angles_to_quat(np.array([np.pi, 0, 0])),
                scale=np.array([0.25, 0.25, 0.25]),
                collision=True
            )
        )
        self._packing_bin_geom = self._world.scene.get_object(f"packing_bin")
        massAPI = UsdPhysics.MassAPI.Apply(self._packing_bin_geom.prim.GetPrim())
        massAPI.CreateMassAttr().Set(0.001)

    def setup_box(self):
        # Box to be picked
        self.box_start_pose = dc.Transform([0.2, 0.08, 0.06], [1, 0, 0, 0])

        self._stage = omni.usd.get_context().get_stage()
        self._boxGeom = createRigidBody(
            self._stage,
            UsdGeom.Cube,
            "/World/Box",
            0.0010,
            [0.015, 0.015, 0.015], 
            self.box_start_pose.p, 
            self.box_start_pose.r, 
            [0.2, 0.2, 1]
        )

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.setup_robot()
        
        # # bin case
        # self.setup_bin()
        
        # box case
        self.setup_box()

        return

    async def setup_post_load(self):
        self._world = self.get_world()

        self._my_controller = PickPlaceController(
            name="controller",
            gripper=self._gripper,
            robot_articulation=self._my_mirobot, 
            events_dt=[
                0.008, 
                0.005, 
                0.1, 
                0.1,
                0.0025, 
                0.001, 
                0.0025, 
                0.5,
                0.008, 
                0.08
            ],
        )
        self._articulation_controller = self._my_mirobot.get_articulation_controller()

        self._world.add_physics_callback("sim_step", callback_fn=self.sim_step_cb)
        return

    async def setup_post_reset(self):
        self._my_controller.reset()
        await self._world.play_async()
        return

    def sim_step_cb(self, step_size):

        # # bin case
        # bin_pose, _ = self._packing_bin_geom.get_world_pose()
        # pick_position = bin_pose
        # place_position = self._target_position

        # box case
        box_matrix = omni.usd.get_world_transform_matrix(self._boxGeom)
        box_trans = box_matrix.ExtractTranslation()
        pick_position = np.array(box_trans)
        place_position = self._target_position

        joints_state = self._my_mirobot.get_joints_state()
        
        actions = self._my_controller.forward(
            picking_position=pick_position,
            placing_position=place_position,
            current_joint_positions=joints_state.positions,
            # This offset needs tuning as well
            end_effector_offset=np.array([0, 0, 0.02947+0.02]),
            end_effector_orientation=euler_angles_to_quat(np.array([0, 0, 0])),
        )
        if self._my_controller.is_done():
            print("done picking and placing")
        self._articulation_controller.apply_action(actions)

        return