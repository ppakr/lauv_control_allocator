# LAUV Control Allocator

A ROS 2 package responsible for mapping desired forces and torques (Wrench) into specific actuator commands (Fin angles and Thruster force) for the LAUV (Light Autonomous Underwater Vehicle).

It uses **CasADi** to solve a non-linear optimization problem, finding the optimal actuator configuration that minimizes tracking error and control effort while respecting physical limits.

### Features
- **Optimization-based Allocation:** Uses CasADi (IPOPT) to solve the allocation problem.
- **Hydrodynamic Fin Model:** Calculates Lift and Drag forces based on flow speed and angle of attack.
- **Physics-Aware:** Validates that vertical fins control Yaw and horizontal fins control Pitch.
- **Safety:** Clamps commands to physical actuator limits.

---
## Node: `lauv_control_allocator`

### Subscribed Topics (Inputs)

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/lauv/wrench_command` | `geometry_msgs/msg/Wrench` | Desired Force (X,Y,Z) and Torque (Roll, Pitch, Yaw). |
| `/lauv/odometry` | `nav_msgs/msg/Odometry` | Current vehicle state. Linear velocity (surge) is required to calculate fin lift. |

### Published Topics (Outputs)

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/model/lauv/joint/fin_X_joint/cmd_pos` | `std_msgs/msg/Float64` | Commanded angle for fins 0-3 (radians). |
| `/model/lauv/joint/thruster_0_joint/cmd_thrust` | `std_msgs/msg/Float64` | Commanded force for the thruster (Newtons). |

### Actuator Configuration

The node is configured for an LAUV with a standard "+" fin configuration:



* **Fin 0 (Top) & Fin 2 (Bottom):** Vertical fins controlling **Yaw**.
* **Fin 1 (Right) & Fin 3 (Left):** Horizontal fins controlling **Pitch**.

---


## Run the node:

```
ros2 run lauv_control_allocator lauv_control_allocator
```


