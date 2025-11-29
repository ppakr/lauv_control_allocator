import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
import tf_transformations

# Message types
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry  # Changed to use standard Odometry

# Import the class you created previously
from lauv_control_allocator.fin_model import FinModel

class LAUVControlAllocator(Node):
    def __init__(self):
        super().__init__('lauv_control_allocator')
        
        # --- 1. Parameter Setup ---
        # Physical constraints
        self.max_fin_angle = np.radians(45.0) # +/- 45 degrees
        self.max_thrust = 50.0                # Max Newtons
        self.min_thrust = -50.0
        
        # Allocation Tuning
        self.control_rate = 0.1  # Run every 0.1s (10Hz)
        
        # State storage
        self.current_velocity = 0.0
        self.target_wrench = np.zeros(6) # [Fx, Fy, Fz, Tx, Ty, Tz]

        # --- 2. Actuator Setup ---
        # We define 4 Fins based on your topic list.
        # NOTE: You may need to verify which ID (0-3) corresponds to Top/Bottom/Port/Starboard
        # for your specific URDF. I am assuming a standard "+" configuration here.
        
        self.fins = []
        
        # Fin 0 (Assume Top Vertical) -> Controls Yaw
        self.fins.append(FinModel(0, np.array([-0.5, 0.0, 0.15]), 
                                  tf_transformations.quaternion_from_euler(1.57, 0, 0), 
                                  "/model/lauv/joint/fin_0_joint/cmd_pos", self))
        
        # Fin 1 (Assume Bottom Vertical) -> Controls Yaw
        self.fins.append(FinModel(1, np.array([-0.5, 0.0, -0.15]), 
                                  tf_transformations.quaternion_from_euler(1.57, 0, 0), 
                                  "/model/lauv/joint/fin_1_joint/cmd_pos", self))
        
        # Fin 2 (Assume Starboard Horizontal) -> Controls Pitch
        self.fins.append(FinModel(2, np.array([-0.5, -0.15, 0.0]), 
                                  np.array([0., 0., 0., 1.]), 
                                  "/model/lauv/joint/fin_2_joint/cmd_pos", self))
        
        # Fin 3 (Assume Port Horizontal) -> Controls Pitch
        self.fins.append(FinModel(3, np.array([-0.5, 0.15, 0.0]), 
                                  np.array([0., 0., 0., 1.]), 
                                  "/model/lauv/joint/fin_3_joint/cmd_pos", self))

        # Thruster Publisher
        self.thruster_pub = self.create_publisher(Float64, 
                                                  '/model/lauv/joint/thruster_0_joint/cmd_thrust', 
                                                  10)

        # --- 3. CasADi Solver Setup ---
        self.setup_solver()

        # --- 4. ROS Subscriptions ---
        # Listen for desired forces/torques (Wrench)
        # Note: This topic won't exist until you write a controller to publish it!
        self.wrench_sub = self.create_subscription(
            Wrench, '/lauv/wrench_command', self.wrench_callback, 10)
        
        # Listen for current velocity from Odometry
        self.odom_sub = self.create_subscription(
            Odometry, '/lauv/odometry', self.velocity_callback, 10)

        # Timer loop
        self.timer = self.create_timer(self.control_rate, self.allocate)
        
        self.get_logger().info("Control Allocator Node Initialized")

    def setup_solver(self):
        """
        Builds the symbolic CasADi optimization problem.
        """
        # [Same logic as before, just ensuring symbols are set up]
        self.u = ca.SX.sym('u', 5) # [fin0, fin1, fin2, fin3, thruster]
        self.p = ca.SX.sym('p', 7) # [Fx_des, ... Tz_des, velocity]
        tau_des = self.p[0:6]
        velocity = self.p[6] 

        total_wrench = ca.SX.zeros(6)
        total_wrench[0] += self.u[4] # Add Thruster Force

        rho = 1000.0 
        fin_area = 0.005 
        
        for i, fin in enumerate(self.fins):
            delta = self.u[i] 
            # Lift = 0.5 * rho * v^2 * Area * CL_slope * angle
            lift_mag = 0.5 * rho * (velocity**2) * fin_area * 5.0 * delta 
            # Drag = 0.5 * rho * v^2 * Area * Cd
            drag_mag = 0.5 * rho * (velocity**2) * fin_area * 0.1

            # Force in Fin Frame (-Drag, Lift, 0)
            f_fin = ca.vertcat(-drag_mag, lift_mag, 0)

            # Rotate to Body Frame
            R_num = fin.R.tolist() 
            R_ca = ca.DM(R_num) 
            f_body = ca.mtimes(R_ca, f_fin)

            # Torque = r x f
            r_ca = ca.DM(fin.pos.tolist())
            t_body = ca.cross(r_ca, f_body)

            total_wrench += ca.vertcat(f_body, t_body)

        # Cost Function
        W = ca.diag([10.0, 10.0, 10.0, 50.0, 50.0, 50.0]) 
        R_reg = ca.diag([1.0, 1.0, 1.0, 1.0, 0.1])        

        error = total_wrench - tau_des
        cost = ca.mtimes([error.T, W, error]) + ca.mtimes([self.u.T, R_reg, self.u])

        nlp = {'x': self.u, 'p': self.p, 'f': cost}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.solver = ca.nlpsol('S', 'ipopt', nlp, opts)

    def wrench_callback(self, msg):
        """Store the desired Wrench (Force/Torque)."""
        self.target_wrench = np.array([
            msg.force.x, msg.force.y, msg.force.z,
            msg.torque.x, msg.torque.y, msg.torque.z
        ])

    def velocity_callback(self, msg):
        """Store current forward speed (u) from Odometry."""
        # Odometry twist is in child_frame_id (usually base_link), so it's already body velocity
        self.current_velocity = msg.twist.twist.linear.x

    def allocate(self):
        """Run the optimization and publish commands."""
        # Safety: If speed is too low, solver might act weird or fins do nothing
        safe_velocity = max(abs(self.current_velocity), 0.1)

        # Prepare Inputs
        p_val = np.concatenate((self.target_wrench, [safe_velocity]))
        
        # Constraints (Bounds)
        lbx = [-self.max_fin_angle]*4 + [self.min_thrust]
        ubx = [self.max_fin_angle]*4 + [self.max_thrust]
        x0 = [0.0] * 5

        # Solve
        try:
            sol = self.solver(x0=x0, p=p_val, lbx=lbx, ubx=ubx)
            u_opt = sol['x'].full().flatten()
            self.publish_control_cmd(u_opt)
        except Exception as e:
            self.get_logger().error(f"Solver failed: {e}")

    def publish_control_cmd(self, u_opt):
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

if __name__ == '__main__':
    main()