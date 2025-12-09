import casadi as ca
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import tf_transformations

from lauv_control_allocator.fin_model import FinModel


class LAUVControlAllocator(Node):
    def __init__(self):
        super().__init__("lauv_control_allocator")

        # --- Parameter Setup ---
        # Physical constraints
        self.max_fin_angle = np.radians(45.0)  # +/- 45 degrees
        self.max_thrust = 50.0  # Max Newtons
        self.min_thrust = -50.0

        # Allocation Tuning
        self.control_rate = 0.1  # Run every 0.1s (10Hz)

        # State storage
        self.current_velocity = 0.0
        self.target_wrench = np.zeros(6)  # [Fx, Fy, Fz, Tx, Ty, Tz]

        # --- Actuator Setup ---
        self.fins = []

        # Geometry Constants (approximate lever arms from center of gravity)
        x_off = -0.5  # Fins are 0.5m behind CoG
        fin_dist = 0.15  # Fins are 15cm from center line

        # fin_0: TOP (Vertical) -> Controls YAW
        q_top = tf_transformations.quaternion_from_euler(0, 0, 0)
        self.fins.append(
            FinModel(
                0,
                np.array([x_off, 0.0, fin_dist]),
                q_top,
                "/model/lauv/joint/fin_0_joint/cmd_pos",
                self,
            )
        )

        # fin_1: RIGHT (Horizontal) -> Controls PITCH
        q_right = tf_transformations.quaternion_from_euler(-1.5708, 0, 0)
        self.fins.append(
            FinModel(
                1,
                np.array([x_off, -fin_dist, 0.0]),
                q_right,
                "/model/lauv/joint/fin_1_joint/cmd_pos",
                self,
            )
        )

        # fin_2: BOTTOM (Vertical) -> Controls YAW
        q_bottom = tf_transformations.quaternion_from_euler(3.14159, 0, 0)
        self.fins.append(
            FinModel(
                2,
                np.array([x_off, 0.0, -fin_dist]),
                q_bottom,
                "/model/lauv/joint/fin_2_joint/cmd_pos",
                self,
            )
        )

        # fin_3: LEFT (Horizontal) -> Controls PITCH
        q_left = tf_transformations.quaternion_from_euler(1.5708, 0, 0)
        self.fins.append(
            FinModel(
                3,
                np.array([x_off, fin_dist, 0.0]),
                q_left,
                "/model/lauv/joint/fin_3_joint/cmd_pos",
                self,
            )
        )

        # Thruster Publisher
        self.thruster_pub = self.create_publisher(
            Float64, "/model/lauv/joint/thruster_0_joint/cmd_thrust", 10
        )

        # --- CasADi Solver Setup ---
        self.setup_solver()

        # --- ROS Subscriptions ---
        self.wrench_sub = self.create_subscription(
            WrenchStamped, "/lauv/wrench_command", self.wrench_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, "/lauv/odometry", self.velocity_callback, 10
        )

        # Timer loop
        self.timer = self.create_timer(self.control_rate, self.allocate)

        self.get_logger().info("Control Allocator Node Initialized")

    def setup_solver(self):
        """Build the symbolic CasADi optimization problem."""
        self.u = ca.SX.sym("u", 5)  # [fin0, fin1, fin2, fin3, thruster]
        self.p = ca.SX.sym("p", 7)  # [Fx_des, ... Tz_des, velocity]
        tau_des = self.p[0:6]
        velocity = self.p[6]

        total_wrench = ca.SX.zeros(6)
        total_wrench[0] += self.u[4]  # Thruster Force X

        rho = 1000.0
        fin_area = 0.005

        for i, fin in enumerate(self.fins):
            delta = self.u[i]
            # Lift & Drag Model
            lift_mag = 0.5 * rho * (velocity**2) * fin_area * 5.0 * delta
            drag_mag = 0.5 * rho * (velocity**2) * fin_area * 0.1

            f_fin = ca.vertcat(-drag_mag, lift_mag, 0)

            # Rotate to Body
            R_num = fin.R.tolist()
            R_ca = ca.DM(R_num)
            f_body = ca.mtimes(R_ca, f_fin)

            # Torque
            r_ca = ca.DM(fin.pos.tolist())
            t_body = ca.cross(r_ca, f_body)
            total_wrench += ca.vertcat(f_body, t_body)

        # Cost Function
        W = ca.diag([10.0, 10.0, 10.0, 50.0, 50.0, 50.0])
        R_reg = ca.diag([1.0, 1.0, 1.0, 1.0, 0.1])

        error = total_wrench - tau_des
        cost = ca.mtimes([error.T, W, error]) + ca.mtimes([self.u.T, R_reg, self.u])

        nlp = {"x": self.u, "p": self.p, "f": cost}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.solver = ca.nlpsol("S", "ipopt", nlp, opts)

    def wrench_callback(self, msg):
        """Store the desired Wrench (Force/Torque)."""
        # DEBUG: Log received wrench
        # self.get_logger().info(f"Received Wrench Command: {msg.wrench}")
        self.target_wrench = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
        )

    def velocity_callback(self, msg):
        """Store current forward speed (u) from Odometry."""
        self.current_velocity = msg.twist.twist.linear.x

    def allocate(self):
        """Run the optimization and publish commands."""
        # Safety: If speed is too low, solver might act weird or fins do nothing
        safe_velocity = self.current_velocity
        if abs(safe_velocity) < 0.1:
            safe_velocity = 1.0

        # check
        if np.linalg.norm(self.target_wrench) < 0.01:
            self.publish_control_cmd(np.zeros(5))
            return

        # Prepare Inputs
        p_val = np.concatenate((self.target_wrench, [safe_velocity]))

        # Constraints (Bounds)
        lbx = [-self.max_fin_angle] * 4 + [self.min_thrust]
        ubx = [self.max_fin_angle] * 4 + [self.max_thrust]

        x0 = [0.0] * 5

        # Solve
        try:
            sol = self.solver(x0=x0, p=p_val, lbx=lbx, ubx=ubx)
            u_opt = sol["x"].full().flatten()

            # --- DEBUG BLOCK (Remove later) ---
            # self.get_logger().info(f"Target Wrench: {self.target_wrench}")
            # self.get_logger().info(
            #     f"Solver Output: Thrust={u_opt[4]:.2f}, Fin0={u_opt[0]:.2f}"
            # )

            # # print out each fin angle
            # for i in range(4):
            #     self.get_logger().info(f"Fin{i} Angle: {u_opt[i]:.2f}")
            # ----------------------------------

            # flip fin angles for correct direction
            # u_opt[0] = -u_opt[0]
            # u_opt[1] = -u_opt[1]
            self.publish_control_cmd(u_opt)
        except Exception as e:
            self.get_logger().error(f"Solver failed: {e}")

    def publish_control_cmd(self, u_opt):
        """Publish the calculated commands to ROS topics."""
        # Publish Fins
        for i, fin in enumerate(self.fins):
            fin.publish_command(u_opt[i])

        # Publish Thruster
        thruster_msg = Float64()
        thruster_msg.data = float(u_opt[4])
        self.thruster_pub.publish(thruster_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LAUVControlAllocator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
