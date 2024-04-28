from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
import omni.isaac.manipulators.controllers as manipulators_controllers
from .rmpflow import RMPFlowController
from omni.isaac.core.articulations import Articulation


# - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
# - Phase 1: Lower end_effector down to encircle the target cube
# - Phase 2: Wait for Robot's inertia to settle.
# - Phase 3: close grip.
# - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
# - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
# - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
# - Phase 7: loosen the grip.
# - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
# - Phase 9: Move end_effector towards the old xy position.

class PickPlaceController(manipulators_controllers.PickPlaceController):
    def __init__(
        self,
        name: str,
        gripper: SurfaceGripper,
        robot_articulation: Articulation,
        events_dt=None
    ) -> None:
        if events_dt is None:
            #These values needs to be tuned in general, you checkout each event in execution and slow it down or speed
            #it up depends on how smooth the movments are
            events_dt = [0.005, 0.002, 1, 0.05, 0.0008, 0.005, 0.0008, 0.1, 0.0008, 0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.05,
            #This value can be changed
            # start_picking_height=0.6
        )
        return