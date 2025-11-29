import unittest

from lauv_control_allocator.fin_model import FinModel
import numpy as np
import rclpy


class TestFinModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize ROS context once for all tests
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        # Shutdown ROS context
        rclpy.shutdown()

    def setUp(self):
        # Create a dummy node to pass to the FinModel
        self.node = rclpy.create_node('test_fin_node_runner')

    def tearDown(self):
        self.node.destroy_node()

    def test_initialization(self):
        """Test if the model initializes with correct parameters."""
        pos = np.array([1.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        fin = FinModel(1, pos, quat, '/test_topic', self.node)

        self.assertEqual(fin.idx, 1)
        self.assertEqual(fin.topic, '/test_topic')
        # Check if R matrix is identity (since quat is identity)
        np.testing.assert_array_equal(fin.R, np.eye(3))

    def test_force_torque_calculation(self):
        """Test the specific math logic (Lift, Drag, Torque)."""
        # 1. Setup
        # Fin at x=1.0, y=0.1 (offset to create torque)
        pos = np.array([1.0, 0.1, 0.0])
        # No rotation
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        fin = FinModel(0, pos, quat, 'fin_topic', self.node)

        # 2. Inputs
        alpha = 0.1  # rad
        Va = 2.0     # m/s

        # 3. Execute
        F, T = fin.get_force_and_torque(alpha, Va)

        # 4. Expected Values (Manually calculated)
        # Lift = 0.1 * 4 * 0.1 = 0.04
        # Drag = 0.01 * 4 * 1.1 = 0.044
        # Force_local = [-0.044, 0.04, 0]
        # Torque = r x F
        # r = [1.0, 0.1, 0]
        # F = [-0.044, 0.04, 0]
        # Tx = 0
        # Ty = 0
        # Tz = (1.0 * 0.04) - (0.1 * -0.044) = 0.04 + 0.0044 = 0.0444

        expected_F = np.array([-0.044, 0.04, 0.0])
        expected_T = np.array([0.0, 0.0, 0.0444])

        # 5. Assert (using approx equality for floating point math)
        np.testing.assert_allclose(
            F, expected_F, rtol=1e-5, err_msg='Force calc incorrect'
        )
        np.testing.assert_allclose(
            T, expected_T, rtol=1e-5, err_msg='Torque calc incorrect'
        )

    def test_zero_speed(self):
        """Ensure no forces are generated at zero speed."""
        pos = np.array([0., 0., 0.])
        quat = np.array([0., 0., 0., 1.])
        fin = FinModel(0, pos, quat, 'topic', self.node)

        F, T = fin.get_force_and_torque(0.5, 0.0)

        np.testing.assert_array_equal(F, np.zeros(3))
        np.testing.assert_array_equal(T, np.zeros(3))