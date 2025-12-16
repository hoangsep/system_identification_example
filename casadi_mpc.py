#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import pandas as pd
import torch
import casadi as ca
import pickle
import time
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt

# ============================
# 1. CONFIGURATION
# ============================
MODEL_PATH = 'gem_dynamics.pth'
SCALER_PATH = 'gem_scaler.pkl'
SCALER_ARRAY_PATH = 'gem_scaler_arrays.npz'
PATH_CSV = 'wps.csv'

# Numpy < 2.0 compatibility
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core

# MPC Settings
HORIZON = 10           # Slightly longer horizon to account for lag
DT = 0.05             # Must match training TARGET_DT
L_WHEELBASE = 1.75
MAX_ACCEL = 2.0
MAX_STEER_RATE = 0.5

# CSV Column Mapping
COL_X = 0
COL_Y = 1
COL_YAW = 2
COL_V = 4

MIN_REF_V = 0.5
MAX_REF_V = 5.5

# Look-ahead tuning
PREVIEW_MIN = 2.0     
PREVIEW_BASE = 2.5
PREVIEW_GAIN = 0.6    
PREVIEW_MAX = 8.0     
PREVIEW_CTE_GAIN = 0.5

class NeuralMPC:
    def __init__(self):
        rospy.init_node('neural_mpc_controller')
        self.current_state = None # [x, y, yaw, v, steer_actual, yaw_rate]
        self.cte_history = []
        self.rate = rospy.Rate(1.0 / DT)
        self.initialized = False
        self.prev_u_opt = None
        self.prev_x_opt = None
        
        # Sensor data holders
        self._pose_data = None
        self._steer_data = 0.0
        self._left_idx = None
        self._right_idx = None

        # --- 1. Load Scalers ---
        if not self.load_scalers():
            return

        # --- 2. Load Neural Network Weights ---
        # Architecture: 6 Input -> 5 Output
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        self.weights = {}
        for key, val in state_dict.items():
            self.weights[key] = val.cpu().numpy()
        print("Model weights loaded (6 inputs -> 5 outputs).")

        # --- 3. Load Path ---
        df = pd.read_csv(PATH_CSV, header=None)
        self.full_path = df.values
        if len(self.full_path) > 1:
            diffs = np.diff(self.full_path[:, [COL_X, COL_Y]], axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            self.path_s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        else:
            self.path_s = np.array([0.0])
        print(f"Path loaded: {len(self.full_path)} points.")

        # --- 4. Setup MPC ---
        self.setup_mpc()

        # --- 5. ROS Setup ---
        self.pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.vis_pub = rospy.Publisher('/gem/mpc_debug', MarkerArray, queue_size=1, latch=True)
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)
        rospy.Subscriber('/gem/joint_states', JointState, self.joint_state_callback)

        self.teleport_to_start()
        print("MPC Controller Ready.")
        self.initialized = True

    def load_scalers(self):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scalers = pickle.load(f)
                self.sx_mean = np.asarray(scalers['x'].mean_, dtype=float)
                self.sx_scale = np.asarray(scalers['x'].scale_, dtype=float)
                self.sy_mean = np.asarray(scalers['y'].mean_, dtype=float)
                self.sy_scale = np.asarray(scalers['y'].scale_, dtype=float)
            return True
        except Exception as e:
            try:
                arrs = np.load(SCALER_ARRAY_PATH)
                self.sx_mean = arrs['x_mean']
                self.sx_scale = arrs['x_scale']
                self.sy_mean = arrs['y_mean']
                self.sy_scale = arrs['y_scale']
                return True
            except Exception:
                rospy.logerr("Failed to load scalers.")
                return False

    def build_initial_guess(self, ref_traj, use_prev=True):
        # State: [x, y, yaw, v, steer_actual]
        curr_x, curr_y, curr_yaw, curr_v, curr_steer, curr_yaw_rate = self.current_state

        if use_prev and self.prev_u_opt is not None:
            u_guess = np.hstack((self.prev_u_opt[:, 1:], self.prev_u_opt[:, -1:]))
        else:
            u_guess = np.zeros((2, HORIZON))
            u_guess[0, :] = curr_v # Constant speed
            # Simple heading P-controller for steering guess
            ref_x, ref_y, _, _ = ref_traj[:, 0]
            heading_err = self.wrap_angle(np.arctan2(ref_y - curr_y, ref_x - curr_x) - curr_yaw)
            u_guess[1, :] = np.clip(heading_err, -0.5, 0.5)

        # Rollout with Numpy Model
        x_guess = np.zeros((5, HORIZON + 1))
        x_guess[:, 0] = [curr_x, curr_y, curr_yaw, curr_v, curr_steer]
        
        temp_state = x_guess[:, 0].copy()
        temp_yaw_rate = curr_yaw_rate # Propagate yaw rate for inputs

        for k in range(HORIZON):
            # rollout_dynamics_np now returns [dx, dy, dyaw, dv, dsteer]
            deltas = self.rollout_dynamics_np(temp_state, u_guess[:, k], temp_yaw_rate)
            
            # Update state
            temp_state = temp_state + deltas
            temp_state[2] = self.wrap_angle(temp_state[2]) # Wrap yaw
            x_guess[:, k + 1] = temp_state
            
            # Estimate next yaw rate (d_yaw / dt) for the next input
            temp_yaw_rate = deltas[2] / DT

        return x_guess, u_guess

    def rollout_dynamics_np(self, state, control, yaw_rate):
        """
        Numpy inference for warm-start generation.
        State: [x, y, yaw, v, steer_actual]
        Control: [cmd_v, cmd_steer]
        Input to NN: [v, steer_actual, yaw_rate, cmd_v, cmd_s, dt]
        """
        # Extract features
        v, steer_act = state[3], state[4]
        cmd_v, cmd_s = control[0], control[1]
        
        inp = np.array([v, steer_act, yaw_rate, cmd_v, cmd_s, DT], dtype=float)
        inp_norm = (inp - self.sx_mean) / self.sx_scale

        # Feedforward
        act = np.tanh
        h1 = act(self.weights['net.0.weight'].dot(inp_norm) + self.weights['net.0.bias'])
        h2 = act(self.weights['net.2.weight'].dot(h1) + self.weights['net.2.bias'])
        h3 = act(self.weights['net.4.weight'].dot(h2) + self.weights['net.4.bias'])
        out_norm = self.weights['net.6.weight'].dot(h3) + self.weights['net.6.bias']
        out_real = out_norm * self.sy_scale + self.sy_mean

        # Output: [dx_local, dy_local, d_yaw, d_v, d_steer]
        yaw_curr = state[2]
        dx_glob = np.cos(yaw_curr) * out_real[0] - np.sin(yaw_curr) * out_real[1]
        dy_glob = np.sin(yaw_curr) * out_real[0] + np.cos(yaw_curr) * out_real[1]
        
        return np.array([dx_glob, dy_glob, out_real[2], out_real[3], out_real[4]], dtype=float)

    def setup_mpc(self):
        opti = ca.Opti()
        self.opti = opti
        
        # --- Variables ---
        # State: [x, y, yaw, v, steer_actual] (5 dims)
        self.X_var = opti.variable(5, HORIZON + 1)
        # Control: [cmd_v, cmd_steer]
        self.U_var = opti.variable(2, HORIZON)
        
        # --- Parameters ---
        self.P_current = opti.parameter(5) # Initial [x,y,yaw,v,steer]
        self.P_yaw_rate_0 = opti.parameter() # Initial yaw rate (from sensor)
        self.P_ref = opti.parameter(4, HORIZON) # Reference [x,y,yaw,v]

        # --- Cost Weights ---
        Q_pos = 60.0    
        Q_yaw = 5.0     
        Q_vel = 10.0 
        R_steer = 1.0    
        R_dsteer = 5.0  # Penalize rapid command changes
        R_accel = 1.0
        Q_steer_state = 0.5 # Regularize actual steering to 0 to prefer straights

        cost = 0
        
        # We need to track yaw_rate inside the loop for the NN input.
        # Initialize with the parameter
        yaw_rate_k = self.P_yaw_rate_0

        for k in range(HORIZON):
            # 1. Cost Calculation
            dx = self.X_var[0, k+1] - self.P_ref[0, k]
            dy = self.X_var[1, k+1] - self.P_ref[1, k]
            e_lat = dx**2 + dy**2 # Simple Euclidean for robustness
            
            yaw_diff = self.X_var[2, k+1] - self.P_ref[2, k]
            e_yaw = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))
            
            e_v = self.X_var[3, k+1] - self.P_ref[3, k]

            cost += Q_pos * e_lat
            cost += Q_yaw * (e_yaw**2)
            cost += Q_vel * (e_v**2)
            cost += R_accel * (self.U_var[0, k]**2)
            cost += R_steer * (self.U_var[1, k]**2)
            cost += Q_steer_state * (self.X_var[4, k+1]**2) # Minimize steer angle

            if k < HORIZON - 1:
                cost += R_dsteer * (self.U_var[1, k+1] - self.U_var[1, k])**2

            # 2. Dynamics Constraint (Neural Net)
            state_k = self.X_var[:, k] # [x,y,yaw,v,steer]
            ctrl_k  = self.U_var[:, k] # [cmd_v, cmd_s]

            # Input: [v, steer_actual, yaw_rate, cmd_v, cmd_s, dt]
            input_features = ca.vertcat(state_k[3], state_k[4], yaw_rate_k, ctrl_k[0], ctrl_k[1], DT)
            inp_norm = (input_features - self.sx_mean) / self.sx_scale
            
            # Forward Pass
            act = ca.tanh
            h1 = act(ca.mtimes(self.weights['net.0.weight'], inp_norm) + self.weights['net.0.bias'])
            h2 = act(ca.mtimes(self.weights['net.2.weight'], h1) + self.weights['net.2.bias'])
            h3 = act(ca.mtimes(self.weights['net.4.weight'], h2) + self.weights['net.4.bias'])
            out_norm = ca.mtimes(self.weights['net.6.weight'], h3) + self.weights['net.6.bias']
            out_real = out_norm * self.sy_scale + self.sy_mean
            
            # Output: [dx, dy, dyaw, dv, dsteer]
            d_yaw_pred = out_real[2]

            # Global update
            yaw_curr = state_k[2]
            dx_glob = ca.cos(yaw_curr)*out_real[0] - ca.sin(yaw_curr)*out_real[1]
            dy_glob = ca.sin(yaw_curr)*out_real[0] + ca.cos(yaw_curr)*out_real[1]
            
            opti.subject_to(self.X_var[0, k+1] == state_k[0] + dx_glob)
            opti.subject_to(self.X_var[1, k+1] == state_k[1] + dy_glob)
            opti.subject_to(self.X_var[2, k+1] == state_k[2] + d_yaw_pred)
            opti.subject_to(self.X_var[3, k+1] == state_k[3] + out_real[3])
            opti.subject_to(self.X_var[4, k+1] == state_k[4] + out_real[4]) # Update actual steer

            # Update yaw_rate for next step input (Simple Euler approx)
            yaw_rate_k = d_yaw_pred / DT

        opti.minimize(cost)

        # --- Constraints ---
        opti.subject_to(self.X_var[:, 0] == self.P_current) # Initial Condition

        for k in range(HORIZON):
            # Actuator Limits
            opti.subject_to(opti.bounded(MIN_REF_V, self.U_var[0, k], MAX_REF_V))
            opti.subject_to(opti.bounded(-0.6, self.U_var[1, k], 0.6))
            
            # Rate Limits
            if k > 0:
                opti.subject_to(opti.bounded(-MAX_ACCEL*DT, self.U_var[0,k]-self.U_var[0,k-1], MAX_ACCEL*DT))
                opti.subject_to(opti.bounded(-MAX_STEER_RATE*DT, self.U_var[1,k]-self.U_var[1,k-1], MAX_STEER_RATE*DT))

        # Solver opts
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 500,
            'ipopt.tol': 1e-1,
        }
        opti.solver('ipopt', opts)

    def solve_with_guess(self, ref_traj, use_prev=True):
        try:
            x_guess, u_guess = self.build_initial_guess(ref_traj, use_prev)
            
            # Pass Parameters
            # P_current: [x,y,yaw,v,steer_actual]
            curr_vals = self.current_state[:5] # Drop yaw_rate for state vector
            self.opti.set_value(self.P_current, curr_vals)
            self.opti.set_value(self.P_yaw_rate_0, self.current_state[5]) # Separate param
            self.opti.set_value(self.P_ref, ref_traj)
            
            self.opti.set_initial(self.X_var, x_guess)
            self.opti.set_initial(self.U_var, u_guess)
            
            return self.opti.solve()
        except Exception as e:
            if use_prev: 
                self.prev_u_opt = None
                self.prev_x_opt = None
            return None

    def state_callback(self, msg):
        try:
            names = msg.name
            idx = names.index("gem")
            p = msg.pose[idx].position
            q = msg.pose[idx].orientation
            v_lin = msg.twist[idx].linear
            w_ang = msg.twist[idx].angular
            
            (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            speed = np.sqrt(v_lin.x**2 + v_lin.y**2)
            
            # Store complete state: [x, y, yaw, v, steer_actual, yaw_rate]
            self.current_state = np.array([p.x, p.y, yaw, speed, self._steer_data, w_ang.z])
            
        except ValueError:
            pass

    def joint_state_callback(self, msg):
        try:
            if self._left_idx is None:
                self._left_idx = msg.name.index("left_steering_hinge_joint")
                self._right_idx = msg.name.index("right_steering_hinge_joint")
            
            l = msg.position[self._left_idx]
            r = msg.position[self._right_idx]
            self._steer_data = (l + r) / 2.0
        except ValueError:
            pass

    # ... [Same Visualization, Teleport, and Helper functions as before] ...
    # (Included implicitly to keep response length manageable, key logic updated above)

    @staticmethod
    def wrap_angle(angle):
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle

    def index_ahead_by_distance(self, start_idx, dist):
        if len(self.path_s) == 0: return 0
        current_s = self.path_s[start_idx]
        target_s = current_s + dist
        idx = np.searchsorted(self.path_s, target_s)
        return min(idx, len(self.path_s)-1)

    def get_reference_trajectory(self):
        if self.current_state is None: return None
        curr_x, curr_y = self.current_state[0], self.current_state[1]
        
        # Nearest Point
        dists = np.linalg.norm(self.full_path[:, [COL_X, COL_Y]] - np.array([curr_x, curr_y]), axis=1)
        nearest_idx = np.argmin(dists)
        self.cte_history.append(dists[nearest_idx])
        
        # Adaptive Lookahead
        v = self.current_state[3]
        lookahead = np.clip(PREVIEW_BASE + PREVIEW_GAIN * v, PREVIEW_MIN, PREVIEW_MAX)
        
        start_idx = self.index_ahead_by_distance(nearest_idx, lookahead)
        ref_traj = []
        
        # Sample points based on Velocity * DT
        step_dist = max(v * DT, 0.5) 
        curr_idx = start_idx
        
        for _ in range(HORIZON):
            row = self.full_path[curr_idx]
            ref_traj.append([row[COL_X], row[COL_Y], row[COL_YAW], row[COL_V]])
            curr_idx = self.index_ahead_by_distance(curr_idx, step_dist)
            
        return np.array(ref_traj).T

    def run(self):
        if not self.initialized: return
        while not rospy.is_shutdown():
            if self.current_state is None:
                self.rate.sleep()
                continue
                
            ref_traj = self.get_reference_trajectory()
            use_warm = (self.prev_u_opt is not None)
            
            sol = self.solve_with_guess(ref_traj, use_warm)
            if sol is None and use_warm:
                sol = self.solve_with_guess(ref_traj, False) # Retry cold
            
            msg = AckermannDrive()
            if sol is not None:
                self.prev_u_opt = sol.value(self.U_var)
                self.prev_x_opt = sol.value(self.X_var)
                
                cmd_v = float(self.prev_u_opt[0, 0])
                cmd_s = float(self.prev_u_opt[1, 0])
                
                msg.speed = cmd_v
                msg.steering_angle = cmd_s
                
                if self.vis_pub.get_num_connections() > 0:
                    self.publish_markers(ref_traj, self.prev_x_opt)
            else:
                print("Solver Failed")
                msg.speed = 0.0
            
            self.pub.publish(msg)
            self.rate.sleep()

    def teleport_to_start(self):
        """Teleports the car to the first waypoint and resets controls"""
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # 1. Get First Waypoint
            start_x = self.full_path[0, COL_X]
            start_y = self.full_path[0, COL_Y]
            start_yaw = self.wrap_angle(float(self.full_path[0, COL_YAW]))
            
            # 2. Prepare Gazebo State Msg
            state_msg = ModelState()
            state_msg.model_name = 'gem' # Make sure this matches!
            state_msg.pose.position.x = start_x
            state_msg.pose.position.y = start_y
            state_msg.pose.position.z = 2.0
            
            q = quaternion_from_euler(0, 0, start_yaw)
            state_msg.pose.orientation.x = q[0]
            state_msg.pose.orientation.y = q[1]
            state_msg.pose.orientation.z = q[2]
            state_msg.pose.orientation.w = q[3]
            
            state_msg.twist.linear.x = 0.0
            state_msg.twist.angular.z = 0.0
            
            # 3. Call Service
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            print(f"Teleported to start: {resp.success}")
            
            # 4. Reset Steering
            reset_msg = AckermannDrive()
            reset_msg.speed = 0.0
            reset_msg.steering_angle = 0.0
            self.pub.publish(reset_msg)
            
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

    def publish_markers(self, ref_traj, pred_traj):
        """Publishes Reference (Green) and Predicted (Blue) paths to RViz in LOCAL frame"""
        if self.current_state is None: return

        height = -0.5

        # --- FIX: Handle 6-element state vector ---
        # Current state: [x, y, yaw, v, steer_actual, yaw_rate]
        # We only need x, y, yaw for the coordinate transform.
        curr_x = self.current_state[0]
        curr_y = self.current_state[1]
        curr_yaw = self.current_state[2]
        
        cos_yaw = np.cos(curr_yaw)
        sin_yaw = np.sin(curr_yaw)

        # Helper to transform global [x,y] to local [x,y]
        def global_to_local(gx, gy):
            dx = gx - curr_x
            dy = gy - curr_y
            lx = cos_yaw * dx + sin_yaw * dy
            ly = -sin_yaw * dx + cos_yaw * dy
            return lx, ly

        marker_array = MarkerArray()
        marker_stamp = rospy.Time(0)

        # 1. Reference Path Marker (Green)
        marker_ref = Marker()
        marker_ref.header.frame_id = "base_link"
        marker_ref.header.stamp = marker_stamp
        marker_ref.ns = "mpc_ref"
        marker_ref.id = 0
        marker_ref.type = Marker.POINTS
        marker_ref.action = Marker.ADD
        marker_ref.pose.orientation.w = 1.0
        marker_ref.frame_locked = True
        marker_ref.scale.x = 0.15
        marker_ref.scale.y = 0.15
        marker_ref.color.a = 1.0
        marker_ref.color.g = 1.0 
        
        # ref_traj shape is (4, HORIZON) -> [x, y, yaw, v]
        for i in range(ref_traj.shape[1]):
            lx, ly = global_to_local(ref_traj[0, i], ref_traj[1, i])
            p = Point()
            p.x = lx
            p.y = ly
            p.z = height
            marker_ref.points.append(p)
        marker_array.markers.append(marker_ref)

        # 2. Predicted Path Marker (Blue)
        marker_pred = Marker()
        marker_pred.header.frame_id = "base_link" 
        marker_pred.header.stamp = marker_stamp
        marker_pred.ns = "mpc_pred"
        marker_pred.id = 1
        marker_pred.type = Marker.SPHERE_LIST
        marker_pred.action = Marker.ADD
        marker_pred.pose.orientation.w = 1.0 
        marker_pred.frame_locked = True
        marker_pred.scale.x = 0.1
        marker_pred.scale.y = 0.1
        marker_pred.scale.z = 0.1 
        marker_pred.color.a = 1.0
        marker_pred.color.b = 1.0 # Blue
        
        # Ensure pred_traj is numpy
        if hasattr(pred_traj, 'full'):
            pred_traj = pred_traj.full()
        pred_traj = np.array(pred_traj)

        # pred_traj shape is (5, HORIZON+1) -> [x, y, yaw, v, steer]
        for i in range(pred_traj.shape[1]):
            lx, ly = global_to_local(pred_traj[0, i], pred_traj[1, i])
            
            if np.isnan(lx) or np.isnan(ly): continue

            p = Point()
            p.x = float(lx)
            p.y = float(ly)
            p.z = -0.5
            marker_pred.points.append(p)
        
        marker_array.markers.append(marker_pred)

        self.vis_pub.publish(marker_array)

    def save_plot(self):
        """Generates the required deliverables upon shutdown"""
        if not hasattr(self, 'cte_history') or not self.cte_history:
            return
            
        plt.figure(figsize=(10,6))
        plt.plot(self.cte_history, label='Cross Track Error')
        plt.axhline(y=1.0, color='r', linestyle='--', label='1m Limit')
        plt.title("Path Tracking Performance (CTE)")
        plt.xlabel("Control Steps")
        plt.ylabel("Error (meters)")
        plt.legend()
        plt.grid(True)
        plt.savefig("cross_track_error.png")
        print("Saved cross_track_error.png")

if __name__ == '__main__':
    controller = NeuralMPC()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        print("Shutting down MPC controller.")
    finally:
        controller.save_plot()
