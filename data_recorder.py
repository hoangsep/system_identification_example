#!/usr/bin/env python3
import rospy
import numpy as np
import pandas as pd
import torch
import casadi as ca
import pickle
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION
# ============================
MODEL_PATH = 'gem_dynamics.pth'
SCALER_PATH = 'gem_scaler.pkl'
PATH_CSV = 'path.csv'
HORIZON = 10           # Look ahead steps
DT = 0.05              # Reduced to 0.05s (20Hz) to match typical training data
TARGET_SPEED = 1.5     # Keep it slow and steady
MAX_SPEED = 2.5
LOOKAHEAD_IDX = 40     # How many CSV points to look ahead for the "Target"

class NeuralMPC:
    def __init__(self):
        rospy.init_node('neural_mpc_robust')
        
        # --- 1. Load Model & Scalers ---
        with open(SCALER_PATH, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_x = scalers['x']
            self.scaler_y = scalers['y']
        
        state_dict = torch.load(MODEL_PATH)
        self.weights = {}
        for key, val in state_dict.items():
            self.weights[key] = val.cpu().numpy()

        # --- 2. Load Path ---
        try:
            df = pd.read_csv(PATH_CSV)
            # Handle cases where CSV has headers or not
            if 'x' in df.columns:
                self.path = df[['x', 'y', 'yaw']].values
            else:
                self.path = df.iloc[:, 0:3].values
        except Exception as e:
            rospy.logerr(f"Path load error: {e}")
            return

        print(f"Path loaded: {len(self.path)} points")

        # --- 3. Setup MPC ---
        self.setup_mpc()

        # --- 4. ROS Connections ---
        self.current_state = None
        self.last_path_idx = 0
        self.cte_history = []
        
        self.cmd_pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDriveStamped, queue_size=1)
        self.vis_pub = rospy.Publisher('/mpc_debug', Marker, queue_size=1)
        self.teleport_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)

        # Teleport to start to ensure good initial condition
        rospy.sleep(1.0)
        self.teleport_to_start()
        
        self.rate = rospy.Rate(1.0 / DT) 
        print("Robust MPC Initialized.")

    def teleport_to_start(self):
        if self.current_state is None:
            return
        
        # Teleport to index 0 of path
        pt = self.path[0]
        msg = ModelState()
        msg.model_name = "gem" # CHECK THIS
        msg.pose.position.x = pt[0]
        msg.pose.position.y = pt[1]
        msg.pose.position.z = 0.3
        q = quaternion_from_euler(0, 0, pt[2])
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.teleport_pub.publish(msg)
        
        # Reset tracker
        self.last_path_idx = 0
        rospy.sleep(0.5)

    def setup_mpc(self):
        opti = ca.Opti()
        self.opti = opti
        
        # Dimensions
        self.X_var = opti.variable(4, HORIZON + 1) # [x, y, yaw, v]
        self.U_var = opti.variable(2, HORIZON)     # [v_cmd, steer_cmd]
        self.P_curr = opti.parameter(4)            # Current State
        self.P_ref = opti.parameter(4, HORIZON)    # Reference Trajectory

        cost = 0
        
        # --- TUNING SECTION ---
        # If car cuts corners -> Increase Q_pos
        # If car oscillates -> Increase R_dsteer, Q_yaw
        Q_pos    = 30.0
        Q_yaw    = 15.0  
        Q_vel    = 5.0
        R_steer  = 1.0
        R_accel  = 1.0
        R_dsteer = 10.0 # Smoothness cost

        for k in range(HORIZON):
            # 1. Position Error
            err_x = self.X_var[0, k+1] - self.P_ref[0, k]
            err_y = self.X_var[1, k+1] - self.P_ref[1, k]
            cost += Q_pos * (err_x**2 + err_y**2)
            
            # 2. Heading Error (ROBUST FIX)
            # Instead of (yaw - ref)^2, use vector alignment 
            # This handles the -pi to pi wrap-around automatically
            yaw_pred = self.X_var[2, k+1]
            yaw_ref  = self.P_ref[2, k]
            # 1 - cos(theta) is approx theta^2/2 for small angles, but safe for large ones
            cost += Q_yaw * 10.0 * (1.0 - ca.cos(yaw_pred - yaw_ref))
            
            # 3. Velocity Error
            cost += Q_vel * (self.X_var[3, k+1] - self.P_ref[3, k])**2
            
            # 4. Control Regularization
            cost += R_steer * (self.U_var[1, k]**2)
            cost += R_accel * (self.U_var[0, k]**2)
            
            # 5. Smoothness
            if k < HORIZON - 1:
                cost += R_dsteer * (self.U_var[1, k+1] - self.U_var[1, k])**2

        opti.minimize(cost)

        # --- DYNAMICS & CONSTRAINTS ---
        opti.subject_to(self.X_var[:, 0] == self.P_curr)

        for k in range(HORIZON):
            # Limits
            opti.subject_to(opti.bounded(-MAX_SPEED, self.U_var[0, k], MAX_SPEED))
            opti.subject_to(opti.bounded(-0.6, self.U_var[1, k], 0.6))
            
            # Neural Net Dynamics
            st = self.X_var[:, k]
            con = self.U_var[:, k]
            
            # Normalize Inputs
            # Input vector: [v_actual, v_cmd, steer_cmd, dt]
            # Note: We hardcode DT here as a float constant for the solver
            inp_raw = ca.vertcat(st[3], con[0], con[1], DT)
            inp_norm = (inp_raw - self.scaler_x.mean_) / self.scaler_x.scale_
            
            # Forward Pass (3 Layers of ReLU)
            # Using basic Matrix Multiplication logic in CasADi
            def dense(x, w_name, b_name):
                return ca.mtimes(self.weights[w_name], x) + self.weights[b_name]
            
            h1 = ca.fmax(0, dense(inp_norm, 'net.0.weight', 'net.0.bias'))
            h2 = ca.fmax(0, dense(h1, 'net.2.weight', 'net.2.bias'))
            h3 = ca.fmax(0, dense(h2, 'net.4.weight', 'net.4.bias'))
            out_norm = dense(h3, 'net.6.weight', 'net.6.bias')
            
            # Denormalize Output
            out_real = out_norm * self.scaler_y.scale_ + self.scaler_y.mean_
            
            dx_loc, dy_loc, dyaw, dv = out_real[0], out_real[1], out_real[2], out_real[3]
            
            # Global Frame Update
            theta = st[2]
            dx_glob = ca.cos(theta)*dx_loc - ca.sin(theta)*dy_loc
            dy_glob = ca.sin(theta)*dx_loc + ca.cos(theta)*dy_loc
            
            opti.subject_to(self.X_var[0, k+1] == st[0] + dx_glob)
            opti.subject_to(self.X_var[1, k+1] == st[1] + dy_glob)
            opti.subject_to(self.X_var[2, k+1] == st[2] + dyaw)
            opti.subject_to(self.X_var[3, k+1] == st[3] + dv)

        # Solver Options (More forgiving)
        opts = {
            'ipopt.print_level': 0, 
            'ipopt.max_iter': 1000, 
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_obj_change_tol': 1e-3,
            'print_time': 0
        }
        opti.solver('ipopt', opts)

    def state_callback(self, msg):
        try:
            # Handle different robot names
            idx = -1
            for name in ['gem', 'polaris', 'car']:
                if name in msg.name:
                    idx = msg.name.index(name)
                    break
            if idx == -1: return
        except: return

        p = msg.pose[idx].position
        q = msg.pose[idx].orientation
        v = msg.twist[idx].linear
        
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        speed = np.sqrt(v.x**2 + v.y**2)
        
        self.current_state = np.array([p.x, p.y, yaw, speed])

    def get_ref_traj(self):
        # Sliding Window Search (Prevents jumping to wrong track segment)
        search_len = 100
        start_i = self.last_path_idx
        end_i = min(len(self.path), start_i + search_len)
        
        # Distance to points in window
        window = self.path[start_i:end_i, :2]
        dists = np.linalg.norm(window - self.current_state[:2], axis=1)
        
        min_idx_window = np.argmin(dists)
        global_idx = start_i + min_idx_window
        
        # Save for next loop
        self.last_path_idx = global_idx
        
        # Save CTE
        self.cte_history.append(dists[min_idx_window])
        
        # Generate Horizon Reference
        refs = []
        for k in range(HORIZON):
            # Advance index based on speed estimate
            # If we go 2 m/s, and dt is 0.05, we move 0.1m per step.
            # If path points are dense (e.g. 0.05m apart), we step index by 2.
            step_dist = TARGET_SPEED * DT 
            # Rough estimate: advance index by 2-3 points per horizon step
            t_idx = min(len(self.path)-1, global_idx + int(k * 2) + 2)
            
            pt = self.path[t_idx]
            refs.append([pt[0], pt[1], pt[2], TARGET_SPEED])
            
        return np.array(refs).T

    def visualize(self, predicted_traj):
        # Publish lines to RViz
        marker = Marker()
        marker.header.frame_id = "map" # OR "world"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0 # Green for Predicted
        
        for i in range(predicted_traj.shape[1]):
            p = Point()
            p.x = predicted_traj[0, i]
            p.y = predicted_traj[1, i]
            marker.points.append(p)
            
        self.vis_pub.publish(marker)

    def run(self):
        while not rospy.is_shutdown():
            if self.current_state is None:
                self.rate.sleep()
                continue
                
            try:
                # 1. Get Ref
                ref = self.get_ref_traj()
                
                # 2. Set Parameters
                self.opti.set_value(self.P_curr, self.current_state)
                self.opti.set_value(self.P_ref, ref)
                
                # Warm start with previous state location (simple heuristic)
                self.opti.set_initial(self.X_var, np.tile(self.current_state.reshape(4,1), (1, HORIZON+1)))

                # 3. Solve
                sol = self.opti.solve()
                
                # 4. Drive
                u = sol.value(self.U_var)
                cmd = AckermannDriveStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.drive.speed = u[0, 0]
                cmd.drive.steering_angle = u[1, 0]
                self.cmd_pub.publish(cmd)
                
                # 5. Visualize
                traj = sol.value(self.X_var)
                self.visualize(traj)

            except Exception as e:
                # Soft Fail: Coast to stop, don't crash script
                # print(f"MPC Warning: {e}")
                cmd = AckermannDriveStamped()
                cmd.drive.speed = 0.0
                self.cmd_pub.publish(cmd)
            
            self.rate.sleep()

    def save(self):
        if self.cte_history:
            plt.plot(self.cte_history)
            plt.title("Cross Track Error")
            plt.savefig("cte_plot.png")

if __name__ == '__main__':
    c = NeuralMPC()
    try:
        c.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        c.save()