#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import types
import pandas as pd
import torch
import casadi as ca
import pickle
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
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

# Compatibility shim: if scalers were pickled with numpy>=2, unpickling on numpy<2
# may look for numpy._core.*. Provide an alias so pickle load succeeds without
# forcing a rebuild of the scalers file.
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core

# MPC Settings
HORIZON = 30           # Look ahead steps (Increased 10->20 for better recovery)
DT = 0.05              # Time step (Must match your data recorder freq, e.g., 20Hz = 0.05)
L_WHEELBASE = 1.75     # Meters (approx for Polaris GEM)

# CSV Column Mapping (Based on your snippet)
COL_X = 0
COL_Y = 1
COL_YAW = 2
# COL_V = 4   # Using the dynamic speed from the CSV
MIN_REF_V = 0.5       # m/s floor to avoid stalling at start
MAX_REF_V = 5.5       # match controller speed limit

class NeuralMPC:
    def __init__(self):
        rospy.init_node('neural_mpc_controller')
        # Initialize defaults so shutdown handlers don't fail if init aborts early
        self.current_state = None
        self.cte_history = []
        self.rate = rospy.Rate(1.0 / DT)  # stay in sync with the model/control timestep
        self.initialized = False
        
        # --- 1. Load Scalers ---
        # We need these to normalize inputs for the neural net inside the solver.
        # Some environments throw docstring-related errors when unpickling sklearn objects,
        # so we provide an ndarray fallback (gem_scaler_arrays.npz).
        if not self.load_scalers():
            return

        # --- 2. Load Neural Network Weights ---
        # We manually load weights to reconstruction the network in CasADi
        # Force CPU load to avoid any GPU dependency inside the container
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        self.weights = {}
        for key, val in state_dict.items():
            self.weights[key] = val.cpu().numpy()
        print("Model weights loaded.")

        # --- 3. Load Path from CSV ---
        # header=None because you said the file just starts with numbers
        df = pd.read_csv(PATH_CSV, header=None)
        self.full_path = df.values # Numpy array
        # Pre-compute average segment length to scale index stepping (protect against ultra-dense points)
        if len(self.full_path) > 1:
            diffs = np.diff(self.full_path[:, [COL_X, COL_Y]], axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            self.avg_path_ds = float(np.clip(np.mean(seg_lens), 1e-4, 10.0))
        else:
            self.avg_path_ds = 0.1
        print(f"Path loaded: {len(self.full_path)} points")

        # --- 3.5 Teleport to Start ---
        # Moved to end of init to ensure publisher exists


        # --- 4. Setup MPC Solver ---
        self.setup_mpc()

        # --- 5. ROS Setup ---
        self.current_state = None # [x, y, yaw, v]
        self.cte_history = []
        
        # Publisher for control commands
        self.pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.vis_pub = rospy.Publisher('/gem/mpc_debug', MarkerArray, queue_size=1)
        
        
        # Subscriber for Ground Truth state (Since no Odom/Map frame exists)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)

        # --- 6. Teleport to Start ---
        self.teleport_to_start()
        
        print("MPC Controller Ready. Waiting for Gazebo state...")
        self.initialized = True

    def load_scalers(self):
        """Load scaler stats from pickle, fallback to npz arrays if pickle fails."""
        try:
            with open(SCALER_PATH, 'rb') as f:
                scalers = pickle.load(f)
                self.sx_mean = np.asarray(scalers['x'].mean_, dtype=float)
                self.sx_scale = np.asarray(scalers['x'].scale_, dtype=float)
                self.sy_mean = np.asarray(scalers['y'].mean_, dtype=float)
                self.sy_scale = np.asarray(scalers['y'].scale_, dtype=float)
            print("Scalers loaded from pickle.")
            return True
        except Exception as e:
            rospy.logwarn(f"Could not load pickle scalers ({e}); trying npz fallback.")
            try:
                arrs = np.load(SCALER_ARRAY_PATH)
                self.sx_mean = arrs['x_mean']
                self.sx_scale = arrs['x_scale']
                self.sy_mean = arrs['y_mean']
                self.sy_scale = arrs['y_scale']
                print(f"Scalers loaded from {SCALER_ARRAY_PATH}.")
                return True
            except Exception as e2:
                rospy.logerr(f"Failed to load scalers from both pickle and npz: {e2}")
                return False

    def setup_mpc(self):
        """Constructs the optimization problem using CasADi"""
        opti = ca.Opti()
        self.opti = opti
        
        # --- Decision Variables ---
        self.X_var = opti.variable(4, HORIZON + 1) # [x, y, yaw, v]
        self.U_var = opti.variable(2, HORIZON)     # [v_cmd, steer_cmd]
        
        # --- Parameters ---
        self.P_current = opti.parameter(4)
        self.P_ref = opti.parameter(4, HORIZON) 

        # --- Cost Function Weights (TUNED) ---
        Q_pos = 120.0   # Lowered vs position-heavy tuning to let speed tracking matter more
        Q_yaw = 40.0
        Q_vel = 100.0   # Strongly track the reference speed
        R_steer = 1.0   
        R_accel = 0.2   # Allow faster accel to reach target speed
        R_dsteer = 2.0  # Moderate smoothness

        cost = 0
        for k in range(HORIZON):
            e_x = self.X_var[0, k+1] - self.P_ref[0, k]
            e_y = self.X_var[1, k+1] - self.P_ref[1, k]
            e_yaw = self.X_var[2, k+1] - self.P_ref[2, k]
            e_v = self.X_var[3, k+1] - self.P_ref[3, k]

            cost += Q_pos * (e_x**2 + e_y**2)
            cost += Q_yaw * (e_yaw**2)
            cost += Q_vel * (e_v**2)
            
            cost += R_accel * (self.U_var[0, k]**2)
            cost += R_steer * (self.U_var[1, k]**2)

            if k < HORIZON - 1:
                cost += R_dsteer * (self.U_var[1, k+1] - self.U_var[1, k])**2

        opti.minimize(cost)

        # --- Constraints ---
        opti.subject_to(self.X_var[:, 0] == self.P_current)

        for k in range(HORIZON):
            # Velocity & Steering Limits
            # Prevent reverse by disallowing negative speed commands
            opti.subject_to(opti.bounded(0.0, self.U_var[0, k], 5.5))
            opti.subject_to(opti.bounded(-0.6, self.U_var[1, k], 0.6))

            # --- Neural Network Dynamics (Reconstruction) ---
            state_k = self.X_var[:, k]
            ctrl_k  = self.U_var[:, k]
            
            # Input: [v_actual, v_cmd, steer_cmd, dt]
            input_features = ca.vertcat(state_k[3], ctrl_k[0], ctrl_k[1], DT)
            inp_norm = (input_features - self.sx_mean) / self.sx_scale
            
            # Use tanh to mirror the training activation (see train_model.py)
            act = ca.tanh
            
            # Layer 1
            w1 = self.weights['net.0.weight']; b1 = self.weights['net.0.bias']
            h1 = act(ca.mtimes(w1, inp_norm) + b1)
            # Layer 2
            w2 = self.weights['net.2.weight']; b2 = self.weights['net.2.bias']
            h2 = act(ca.mtimes(w2, h1) + b2)
            # Layer 3
            w3 = self.weights['net.4.weight']; b3 = self.weights['net.4.bias']
            h3 = act(ca.mtimes(w3, h2) + b3)
            # Output Layer
            w4 = self.weights['net.6.weight']; b4 = self.weights['net.6.bias']
            out_norm = ca.mtimes(w4, h3) + b4
            
            out_real = out_norm * self.sy_scale + self.sy_mean
            
            # Global Frame Update
            yaw_curr = state_k[2]
            dx_glob = ca.cos(yaw_curr)*out_real[0] - ca.sin(yaw_curr)*out_real[1]
            dy_glob = ca.sin(yaw_curr)*out_real[0] + ca.cos(yaw_curr)*out_real[1]
            
            opti.subject_to(self.X_var[0, k+1] == state_k[0] + dx_glob)
            opti.subject_to(self.X_var[1, k+1] == state_k[1] + dy_glob)
            opti.subject_to(self.X_var[2, k+1] == state_k[2] + out_real[2])
            opti.subject_to(self.X_var[3, k+1] == state_k[3] + out_real[3])

        # --- SOLVER SETTINGS (CRITICAL UPDATE) ---
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 500,           # <--- CHANGED: Increased 100->500 to help L-BFGS cleanup
            'ipopt.hessian_approximation': 'limited-memory', # <--- FAST SOLVER MODE (L-BFGS)
            # 'ipopt.accept_after_max_steps': 'yes', 
            'ipopt.tol': 1e-2,               # <--- Loosened tolerance (1e-3 -> 1e-2) for faster convergence
            'ipopt.warm_start_init_point': 'yes'
        }
        opti.solver('ipopt', opts)

    def state_callback(self, msg):
        """Extracts robot state from Gazebo ModelStates"""
        try:
            # We look for the model named 'gem' or 'polaris'
            # If your simulation uses a different name, change it here!
            names = msg.name
            if "gem" in names:
                idx = names.index("gem")
            elif "polaris" in names:
                idx = names.index("polaris")
            else:
                return # Robot not found yet

            p = msg.pose[idx].position
            q = msg.pose[idx].orientation
            v_linear = msg.twist[idx].linear
            
            # Convert Quaternion to Yaw
            (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            # Calculate speed magnitude
            speed = np.sqrt(v_linear.x**2 + v_linear.y**2)
            
            # Update state [x, y, yaw, v]
            self.current_state = np.array([p.x, p.y, yaw, speed])
            
        except ValueError:
            print("Model 'gem' or 'polaris' not found in Gazebo ModelStates.")

    def teleport_to_start(self):
        """Teleports the car to the first waypoint and resets controls"""
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # 1. Get First Waypoint
            start_x = self.full_path[0, COL_X]
            start_y = self.full_path[0, COL_Y]
            start_yaw = self.full_path[0, COL_YAW]
            
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

    def get_reference_trajectory(self):
        """Finds the nearest point on the CSV path and slices the next N points"""
        if self.current_state is None:
            return None

        curr_x, curr_y = self.current_state[0], self.current_state[1]

        # 1. Find the nearest point index via Euclidean distance
        # We only check x/y (cols 0 and 1)
        path_xy = self.full_path[:, [COL_X, COL_Y]]
        dists = np.linalg.norm(path_xy - np.array([curr_x, curr_y]), axis=1)
        nearest_idx = np.argmin(dists)

        # Calculate Cross Track Error (for plotting/deliverable)
        cte = dists[nearest_idx]
        self.cte_history.append(cte)

        # 2. Slice the path for the Horizon using DISTANCE based lookahead
        # (Fixes issue where dense points at standstill cause microscopic horizon)
        ref_traj = []
        last_idx = nearest_idx
        
        # We want roughly v*DT distance between points, but at least MIN_DIST
        # so we always see ahead. Use avg_path_ds to convert distance to index
        # steps; cap the step to avoid jumping to the end on very dense paths.
        current_v = self.current_state[3]
        target_step_dist = max(current_v * DT, 0.2) # Minimum 0.2m spacing (4m horizon at 20 steps)
        avg_ds = max(1e-4, getattr(self, "avg_path_ds", 0.1))
        max_step_points = 50  # prevent running to path end when points are extremely dense
        
        for i in range(HORIZON):
            # 1. Start with the nearest/last point
            if i == 0:
                idx = nearest_idx
            else:
                step_points = int(np.clip(round(target_step_dist / avg_ds), 1, max_step_points))
                idx = min(last_idx + step_points, len(self.full_path) - 1)
                last_idx = idx

            # Handle end of path
            if idx >= len(self.full_path):
                idx = len(self.full_path) - 1

            row = self.full_path[idx]
            
            # Extract [x, y, yaw, v]
            ref_x = row[COL_X]
            ref_y = row[COL_Y]
            ref_yaw = row[COL_YAW]
            ref_v = 1.0
            # ref_v = float(np.clip(row[COL_V], MIN_REF_V, MAX_REF_V))
            # print(f"Ref Point {i}: ({ref_x:.2f}, {ref_y:.2f}), Yaw: {ref_yaw:.2f}, V: {ref_v:.2f}")
            
            ref_traj.append([ref_x, ref_y, ref_yaw, ref_v])

        # print(f"nearest_idx: {nearest_idx}, ref: {ref_traj[0]}, CTE: {cte:.2f}m")

        # Return shape: (4, HORIZON)
        return np.array(ref_traj).T

    def run(self):
        if not self.initialized:
            rospy.logerr("NeuralMPC failed to initialize; aborting run().")
            return
        # Warm start memory
        self.prev_u_opt = np.zeros((2, HORIZON))
        self.prev_x_opt = np.zeros((4, HORIZON + 1))
        
        last_u = [0.0, 0.0] # Store last control for fallback

        
        while not rospy.is_shutdown():
            if self.current_state is None:
                self.rate.sleep()
                continue
                
            try:
                ref_traj = self.get_reference_trajectory()
                if ref_traj is None: continue

                self.opti.set_value(self.P_current, self.current_state)
                self.opti.set_value(self.P_ref, ref_traj)

                # --- IMPROVED WARM START ---
                # Check if we have a valid previous solution to shift
                # If we are starting or recovering, we might want to use the linear projection guess
                
                # Shift previous solution for warm start
                # u_prev: [u0, u1, ..., uN-1] -> [u1, ..., uN-1, uN-1]
                u_guess = np.hstack((self.prev_u_opt[:, 1:], self.prev_u_opt[:, -1:]))
                
                # x_prev: [x0, x1, ..., xN] -> [x1, ..., xN, xN] (approximately)
                # But actually, simpler to just rely on re-solving or linear guess for X if U is good.
                # Let's try blending: Use shifted U guess, but use current state physics for X guess.
                
                # Basic Linear Physics Guess for X (same as before, but good for reliable initialization)
                curr_x, curr_y, curr_yaw, curr_v = self.current_state
                x_guess = np.zeros((4, HORIZON + 1))
                for k in range(HORIZON + 1):
                    dist = curr_v * DT * k
                    x_guess[0, k] = curr_x + dist * np.cos(curr_yaw)
                    x_guess[1, k] = curr_y + dist * np.sin(curr_yaw)
                    x_guess[2, k] = curr_yaw
                    x_guess[3, k] = curr_v
                
                self.opti.set_initial(self.X_var, x_guess) # Keeping this safe guess
                self.opti.set_initial(self.U_var, u_guess) # <--- CRITICAL: Use shifted previous controls!

                # Solve
                sol = self.opti.solve()
                
                u_opt = sol.value(self.U_var)
                x_opt = sol.value(self.X_var)
                
                # Store for next iteration
                self.prev_u_opt = u_opt
                self.prev_x_opt = x_opt
                
                cmd_v = u_opt[0, 0]
                cmd_s = u_opt[1, 0]
                
                # Update last valid control
                last_u = [cmd_v, cmd_s]

                msg = AckermannDrive()
                # msg.header.stamp = rospy.Time.now()
                msg.speed = float(cmd_v)
                msg.steering_angle = float(cmd_s)
                self.pub.publish(msg)

            except Exception as e:
                # If solver still fails, use the LAST KNOWN GOOD CONTROL
                # This keeps the car moving rather than freezing
                print(f"Solver Limit/Fail: {e} -> Using Fallback")
                
                # Often 'Opti' object still has the debug values even if it failed
                try:
                    debug_u = self.opti.debug.value(self.U_var)
                    debug_x = self.opti.debug.value(self.X_var)
                    self.prev_x_opt = np.array(debug_x) # <--- Force numpy array
                    
                    cmd_v = float(debug_u[0, 0])
                    cmd_s = float(debug_u[1, 0])
                except:
                    cmd_v, cmd_s = last_u # Absolute fallback
                
                msg = AckermannDrive()
                msg.speed = float(cmd_v)
                msg.steering_angle = float(cmd_s)
                self.pub.publish(msg)
            
            # Visualize
            try:
                self.publish_markers(ref_traj, self.prev_x_opt)
            except Exception as e:
                print(f"Viz Error: {e}")

            self.rate.sleep()

    def publish_markers(self, ref_traj, pred_traj):
        """Publishes Reference (Green) and Predicted (Blue) paths to RViz in LOCAL frame"""
        if self.current_state is None: return

        # Transform to Local Frame (base_link)
        curr_x, curr_y, curr_yaw, _ = self.current_state
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

        # 1. Reference Path Marker
        marker_ref = Marker()
        marker_ref.header.frame_id = "base_link" # <--- CHANGED: Local frame
        marker_ref.header.stamp = rospy.Time.now()
        marker_ref.ns = "mpc_ref"
        marker_ref.id = 0
        marker_ref.type = Marker.LINE_STRIP
        marker_ref.action = Marker.ADD
        marker_ref.pose.orientation.w = 1.0 # <--- Fix: Initialize quaternion
        marker_ref.scale.x = 0.1 
        marker_ref.color.a = 1.0
        marker_ref.color.g = 1.0 # Green
        
        for i in range(ref_traj.shape[1]):
            lx, ly = global_to_local(ref_traj[0, i], ref_traj[1, i])
            p = Point()
            p.x = lx
            p.y = ly
            p.z = 2.0
            marker_ref.points.append(p)
        marker_array.markers.append(marker_ref)

        # 2. Predicted Path Marker
        marker_pred = Marker()
        marker_pred.header.frame_id = "base_link" 
        marker_pred.header.stamp = rospy.Time.now()
        marker_pred.ns = "mpc_pred"
        marker_pred.id = 1
        marker_pred.type = Marker.SPHERE_LIST
        marker_pred.action = Marker.ADD
        marker_pred.pose.orientation.w = 1.0 
        marker_pred.scale.x = 0.1
        marker_pred.scale.y = 0.1
        marker_pred.scale.z = 0.1 
        marker_pred.color.a = 1.0
        marker_pred.color.r = 1.0 # Red
        marker_pred.color.g = 0.0 # Green
        marker_pred.color.b = 0.0 # Blue
        
        # Ensure pred_traj is numpy (handle if it's CasADi DM)
        if hasattr(pred_traj, 'full'):
            pred_traj = pred_traj.full()
        pred_traj = np.array(pred_traj)

        for i in range(pred_traj.shape[1]):
            lx, ly = global_to_local(pred_traj[0, i], pred_traj[1, i])
            
            # Check for NaNs
            if np.isnan(lx) or np.isnan(ly):
                print(f"Skipping NaN predicted point at index {i}")
                continue
            
            # if i < 3: # Debug first few points
            #     print(f"DEBUG: Blue Point {i} Local: ({lx:.2f}, {ly:.2f})")

            p = Point()
            p.x = float(lx)
            p.y = float(ly)
            p.z = 2.0
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
