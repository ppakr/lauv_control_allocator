import unittest
import numpy as np
import rclpy
import tf_transformations
from lauv_control_allocator.fin_model import FinModel

class TestFinPhysics(unittest.TestCase):
    """
    Physically validates that the fin configuration produces 
    the correct control forces (Yaw vs Pitch).
    """

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_physics')
        
        # Common parameters
        self.x_off = -0.5
        self.fin_dist = 0.15
        self.q_identity = np.array([0., 0., 0., 1.])
        self.q_roll_90 = tf_transformations.quaternion_from_euler(1.57, 0, 0)

    def tearDown(self):
        self.node.destroy_node()

    def test_fin0_top_yaw(self):
        """Fin 0 (Top) should produce YAW torque as its primary output."""
        # Setup Fin 0 (Top, Vertical)
        pos = np.array([self.x_off, 0.0, self.fin_dist])
        fin = FinModel(0, pos, self.q_identity, "topic", self.node)
        
        # Apply positive angle
        F, T = fin.get_force_and_torque(alpha=0.2, Va=2.0)
        
        print(f"\nFin 0 (Top) Torque: {T}")
        
        # 1. Check Primary Control Axis (Yaw / Z)
        self.assertNotEqual(T[2], 0.0, msg="Vertical fin MUST produce Yaw torque")
        
        # 2. Check that Primary Axis is stronger than Parasitic Drag Axis (Pitch / Y)
        # Drag creates a small Pitch moment, but Lift (Yaw) should be stronger.
        yaw_torque = abs(T[2])
        pitch_torque = abs(T[1])
        
        self.assertTrue(yaw_torque > pitch_torque, 
                        f"Yaw ({yaw_torque}) should be dominant over Pitch ({pitch_torque})")

    def test_fin1_right_pitch(self):
        """Fin 1 (Right) should produce PITCH torque as its primary output."""
        # Setup Fin 1 (Right, Horizontal)
        pos = np.array([self.x_off, -self.fin_dist, 0.0])
        fin = FinModel(1, pos, self.q_roll_90, "topic", self.node)
        
        F, T = fin.get_force_and_torque(alpha=0.2, Va=2.0)
        
        print(f"Fin 1 (Right) Torque: {T}")

        # 1. Check Primary Control Axis (Pitch / Y)
        self.assertNotEqual(T[1], 0.0, msg="Horizontal fin MUST produce Pitch torque")

        # 2. Check that Primary Axis is stronger than Parasitic Drag Axis (Yaw / Z)
        pitch_torque = abs(T[1])
        yaw_torque = abs(T[2])

        self.assertTrue(pitch_torque > yaw_torque,
                        f"Pitch ({pitch_torque}) should be dominant over Yaw ({yaw_torque})")

    def test_fin3_left_pitch(self):
        """Fin 3 (Left) should produce PITCH torque as its primary output."""
        # Setup Fin 3 (Left, Horizontal)
        pos = np.array([self.x_off, self.fin_dist, 0.0])
        fin = FinModel(3, pos, self.q_roll_90, "topic", self.node)
        
        F, T = fin.get_force_and_torque(alpha=0.2, Va=2.0)
        
        print(f"Fin 3 (Left) Torque: {T}")

        # 1. Check Primary Control Axis (Pitch / Y)
        self.assertNotEqual(T[1], 0.0)

        # 2. Check Dominance
        pitch_torque = abs(T[1])
        yaw_torque = abs(T[2])
        self.assertTrue(pitch_torque > yaw_torque)