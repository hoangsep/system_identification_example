#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import types
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

# Compatibility shim: if scalers were pickled with numpy>=2, unpickling on numpy<2
# may look for numpy._core.*. Provide an alias so pickle load succeeds without
# forcing a rebuild of the scalers file.
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core

# MPC Settings
HORIZON = 8           # Keep horizon small for faster solve
PUBLISH_DEBUG_MARKERS = True  # Set True to visualize (adds overhead)
DT = 0.1              # Time step (Must match your data recorder freq, e.g., 20Hz = 0.05)
L_WHEELBASE = 1.75     # Meters (approx for Polaris GEM)
MAX_ACCEL = 2.0        # m/s^2 limit on commanded speed change
MAX_STEER_RATE = 0.5   # rad/s limit on steering command rate

# CSV Column Mapping (Based on your snippet)
COL_X = 0
COL_Y = 1
COL_YAW = 2
COL_V = 4   # Using the dynamic speed from the CSV
MIN_REF_V = 0.5       # m/s floor to avoid stalling at start
MAX_REF_V = 5.5       # match controller speed limit

# Look-ahead tuning: shift the target path slightly ahead of the car
# Fix: Lookahead MUST be > Wheelbase (1.75m) for stability.
PREVIEW_MIN = 1.5     
PREVIEW_BASE = 2.0    # Increased from 0.3 to stabilize (Pure Pursuit rule: L > 1.2*Wheelbase)
PREVIEW_GAIN = 0.5    # Scale with speed
PREVIEW_MAX = 5.0     
PREVIEW_CTE_GAIN = 0.5 # Reduced aggression on large errors

class NeuralMPC:
    def __init__(self):
        rospy.init_node('neural_mpc_controller')
        # Initialize defaults so shutdown handlers don't fail if init aborts early
        self.current_state = None
        self.cte_history = []
        self.rate = rospy.Rate(1.0 / DT)  # stay in sync with the model/control timestep
        self.initialized = False
        self.prev_u_opt = None
        self.prev_x_opt = None
        self.steer_actual = float('nan')
        self._left_idx = None
        self._right_idx = None
        
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
        if len(self.full_path) > 1:
            diffs = np.diff(self.full_path[:, [COL_X, COL_Y]], axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            self.path_s = np.concatenate(([0.0], np.cumsum(seg_lens)))
            self.avg_path_ds = float(np.clip(np.mean(seg_lens), 1e-4, 10.0))
            self.total_path_length = float(self.path_s[-1])
        else:
            self.path_s = np.array([0.0], dtype=float)
            self.avg_path_ds = 0.1
            self.total_path_length = 0.0
        print(f"Path loaded: {len(self.full_path)} points (length {self.total_path_length:.2f} m)")


        # --- 4. Setup MPC Solver ---
        self.setup_mpc()

        # --- 5. ROS Setup ---
        self.current_state = None # [x, y, yaw, v]
        self.cte_history = []
        
        # Publisher for control commands
        self.pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        # Latch debug markers so RViz doesn't briefly clear them if a message is dropped
        self.vis_pub = rospy.Publisher('/gem/mpc_debug', MarkerArray, queue_size=5, latch=True)
        
        
        # Subscriber for Ground Truth state (Since no Odom/Map frame exists)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)
        rospy.Subscriber('/gem/joint_states', JointState, self.joint_state_callback)

        # --- 6. Teleport to Start ---
        self.teleport_to_start()
        # pub 0 speed and steering angle
        self.pub.publish(AckermannDrive(speed=0.0, steering_angle=0.0))
        
        print("MPC Controller Ready. Waiting for Gazebo state...")
        self.initialized = True

    def build_initial_guess(self, ref_traj, use_prev=True):
        """Build initial guesses for X and U (warm start or simple heading-based seed)."""
        curr_x, curr_y, curr_yaw, curr_v = self.current_state

        # Controls: shift previous if available
        if use_prev and self.prev_u_opt is not None and np.all(np.isfinite(self.prev_u_opt)):
            u_guess = np.hstack((self.prev_u_opt[:, 1:], self.prev_u_opt[:, -1:]))
        else:
            # Simple seed: aim at the first reference point with a steady speed
            ref_x, ref_y, _, ref_v = ref_traj[:, 0]
            heading_to_ref = np.arctan2(ref_y - curr_y, ref_x - curr_x)
            heading_err = self.wrap_angle(heading_to_ref - curr_yaw)
            steer_guess = float(np.clip(heading_err, -0.4, 0.4))
            speed_guess = float(np.clip(ref_v if np.isfinite(ref_v) else curr_v, MIN_REF_V, MAX_REF_V))
            u_guess = np.zeros((2, HORIZON))
            u_guess[0, :] = speed_guess
            u_guess[1, :] = steer_guess

        # States: re-rollout from the CURRENT state using the guessed controls.
        # This keeps the initial guess dynamically consistent instead of simply
        # shifting the previous predicted states (which can drift and hurt IPOPT).
        x_guess = np.zeros((4, HORIZON + 1))
        x_guess[:, 0] = [curr_x, curr_y, curr_yaw, curr_v]
        steer_actual_seq = self.estimate_steering_profile(u_guess)
        temp_state = np.array([curr_x, curr_y, curr_yaw, curr_v], dtype=float)
        for k in range(HORIZON):
            dx = self.rollout_dynamics_np(temp_state, u_guess[:, k], steer_actual_seq[k])
            temp_state = temp_state + dx
            temp_state[2] = self.wrap_angle(temp_state[2])
            x_guess[:, k + 1] = temp_state

        # Guard against NaNs/Infs from the warm rollout
        if not np.all(np.isfinite(x_guess)):
            x_guess = np.zeros((4, HORIZON + 1))
            for k in range(HORIZON + 1):
                dist = curr_v * DT * k
                x_guess[0, k] = curr_x + dist * np.cos(curr_yaw)
                x_guess[1, k] = curr_y + dist * np.sin(curr_yaw)
                x_guess[2, k] = curr_yaw
                x_guess[3, k] = curr_v

        return x_guess, u_guess

    @staticmethod
    def wrap_angle(angle):
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def wrap_angle_casadi(angle):
        """CasADi-safe angle wrap to [-pi, pi] using atan2."""
        return ca.atan2(ca.sin(angle), ca.cos(angle))

    def index_ahead_by_distance(self, start_idx, distance):
        """Return the path index at least `distance` meters ahead of start_idx."""
        if len(self.path_s) == 0:
            return 0
        start_idx = int(np.clip(start_idx, 0, len(self.path_s) - 1))
        if distance <= 0.0 or start_idx >= len(self.path_s) - 1:
            return start_idx
        target_s = self.path_s[start_idx] + distance
        idx = int(np.searchsorted(self.path_s, target_s, side='left'))
        return min(idx, len(self.path_s) - 1)

    def estimate_steering_profile(self, u_seq):
        """Estimate steer_actual over the horizon using measured step-0 and commanded changes."""
        steer_cmds = u_seq[1, :]
        steer_actual_est = np.zeros(HORIZON, dtype=float)
        steer_actual_est[0] = self.steer_actual if np.isfinite(self.steer_actual) else steer_cmds[0]
        for k in range(1, HORIZON):
            steer_actual_est[k] = steer_cmds[k-1]  # assume actuator reaches previous command by next step
        return steer_actual_est

    def rollout_dynamics_np(self, state, control, steer_actual):
        """Roll the learned dynamics once in numpy for building warm starts."""
        v_actual, v_cmd, steer_cmd = state[3], control[0], control[1]
        inp = np.array([v_actual, v_cmd, steer_cmd, steer_actual, DT], dtype=float)
        inp_norm = (inp - self.sx_mean) / self.sx_scale

        act = np.tanh
        h1 = act(self.weights['net.0.weight'].dot(inp_norm) + self.weights['net.0.bias'])
        h2 = act(self.weights['net.2.weight'].dot(h1) + self.weights['net.2.bias'])
        h3 = act(self.weights['net.4.weight'].dot(h2) + self.weights['net.4.bias'])
        out_norm = self.weights['net.6.weight'].dot(h3) + self.weights['net.6.bias']
        out_real = out_norm * self.sy_scale + self.sy_mean

        yaw_curr = state[2]
        dx_glob = np.cos(yaw_curr) * out_real[0] - np.sin(yaw_curr) * out_real[1]
        dy_glob = np.sin(yaw_curr) * out_real[0] + np.cos(yaw_curr) * out_real[1]
        return np.array([dx_glob, dy_glob, out_real[2], out_real[3]], dtype=float)

    def solve_with_guess(self, ref_traj, use_prev=True):
        """Try solving with either a warm start (use_prev=True) or cold start."""
        try:
            x_guess, u_guess = self.build_initial_guess(ref_traj, use_prev=use_prev)
            self.opti.set_value(self.P_current, self.current_state)
            self.opti.set_value(self.P_ref, ref_traj)
            steer_act_val = self.steer_actual if np.isfinite(self.steer_actual) else float(u_guess[1, 0])
            self.opti.set_value(self.P_steer_actual, steer_act_val)
            self.opti.set_initial(self.X_var, x_guess)
            self.opti.set_initial(self.U_var, u_guess)
            return self.opti.solve()
        except Exception as e:
            if use_prev:
                # Clear the cached warm start so the next iteration does not keep failing.
                self.prev_u_opt = None
                self.prev_x_opt = None
                print(f"Solver warm-start failed ({e}); trying cold start.")
            return None

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
        self.P_current = opti.parameter(4) # [x, y, yaw, v]
        self.P_ref = opti.parameter(4, HORIZON) # [x, y, yaw, v]
        self.P_steer_actual = opti.parameter() # measured steering angle at current step

        # --- Cost Function Weights (TUNED) ---
        Q_lat = 40.0    
        Q_yaw = 1.0     
        Q_vel = 5.0 
        R_steer = 1.0    
        R_accel = 1.0 
        R_dsteer = 1.0 
        Q_safety = 1.0   

        cost = 0
        for k in range(HORIZON):
            dx = self.X_var[0, k+1] - self.P_ref[0, k]
            dy = self.X_var[1, k+1] - self.P_ref[1, k]
            ref_yaw = self.P_ref[2, k]
            e_lat = -ca.sin(ref_yaw) * dx + ca.cos(ref_yaw) * dy
            yaw_diff = self.X_var[2, k+1] - self.P_ref[2, k]
            e_yaw = self.wrap_angle_casadi(yaw_diff)
            e_v = self.X_var[3, k+1] - self.P_ref[3, k]

            cost += Q_lat * (e_lat**2)
            cost += Q_yaw * (e_yaw**2)
            cost += Q_vel * (e_v**2)
            
            # Control Regularization
            cost += R_accel * (self.U_var[0, k]**2)
            cost += R_steer * (self.U_var[1, k]**2)

            # --- SAFETY COST: Slow down when steering ---
            # Penalize (velocity * steering_angle)^2
            # This encourages low velocity if steering is high, or low steering if velocity is high
            cost += Q_safety * (self.U_var[0, k] * self.U_var[1, k])**2

            # if k < HORIZON - 1:
            #     # Damping: Penalize rapid changes in steering
            #     cost += R_dsteer * (self.U_var[1, k+1] - self.U_var[1, k])**2

        opti.minimize(cost)

        # --- Constraints ---
        opti.subject_to(self.X_var[:, 0] == self.P_current)

        for k in range(HORIZON):
            # Velocity & Steering Limits
            # Prevent reverse by disallowing negative speed commands
            opti.subject_to(opti.bounded(MIN_REF_V, self.U_var[0, k], MAX_REF_V))
            opti.subject_to(opti.bounded(-0.6, self.U_var[1, k], 0.6))
            if k > 0:
                # Rate limits derived from recorded logs (per-step change)
                opti.subject_to(opti.bounded(-MAX_ACCEL * DT,
                                             self.U_var[0, k] - self.U_var[0, k-1],
                                             MAX_ACCEL * DT))
                opti.subject_to(opti.bounded(-MAX_STEER_RATE * DT,
                                             self.U_var[1, k] - self.U_var[1, k-1],
                                             MAX_STEER_RATE * DT))

            # --- Neural Network Dynamics (Reconstruction) ---
            state_k = self.X_var[:, k]
            ctrl_k  = self.U_var[:, k]

            # Estimate steering actual: use measured for k=0, otherwise assume actuator hits previous command
            if k == 0:
                steer_actual_k = self.P_steer_actual
            else:
                steer_actual_k = self.U_var[1, k-1]

            # Input: [v_actual, v_cmd, steer_cmd, steer_actual, dt]
            input_features = ca.vertcat(state_k[3], ctrl_k[0], ctrl_k[1], steer_actual_k, DT)
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
            
            # --- MODEL CORRECTION GAIN ---
            # Validation showed the trained model underestimates Yaw Rate by ~6x.
            # Since we cannot retrain, we manually boost the predicted yaw change.
            # This makes the MPC realize the car turns fast, so it commands smoother/smaller steering.
            YAW_GAIN = 6.0 
            out_real[2] = out_real[2] * YAW_GAIN
            
            # Global Frame Update
            yaw_curr = state_k[2]
            dx_glob = ca.cos(yaw_curr)*out_real[0] - ca.sin(yaw_curr)*out_real[1]
            dy_glob = ca.sin(yaw_curr)*out_real[0] + ca.cos(yaw_curr)*out_real[1]
            
            opti.subject_to(self.X_var[0, k+1] == state_k[0] + dx_glob)
            opti.subject_to(self.X_var[1, k+1] == state_k[1] + dy_glob)
            opti.subject_to(self.X_var[2, k+1] == state_k[2] + out_real[2])
            opti.subject_to(self.X_var[3, k+1] == state_k[3] + out_real[3])

        # --- SOLVER SETTINGS ---
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 1000,         
            'ipopt.max_cpu_time': 1.0,     # seconds (avoid premature timeout)
            'ipopt.hessian_approximation': 'limited-memory', # L-BFGS
            'ipopt.tol': 1e-1,
            'ipopt.acceptable_tol': 5e-1,
            'ipopt.acceptable_obj_change_tol': 1e-2,
            'ipopt.warm_start_init_point': 'yes'
        }
        opti.solver('ipopt', opts)

    def state_callback(self, msg):
        """Extracts robot state from Gazebo ModelStates"""
        try:
            # We look for the model named 'gem' 
            names = msg.name
            idx = names.index("gem")

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

    def joint_state_callback(self, msg: JointState):
        """Track measured steering angle and rate from joint states."""
        try:
            if self._left_idx is None:
                self._left_idx = msg.name.index("left_steering_hinge_joint")
            if self._right_idx is None:
                self._right_idx = msg.name.index("right_steering_hinge_joint")
        except ValueError:
            return

        # Protect against short velocity arrays (some publishers omit velocity)
        if self._left_idx >= len(msg.position) or self._right_idx >= len(msg.position):
            return
        left_theta = msg.position[self._left_idx]
        right_theta = msg.position[self._right_idx]

        self.steer_actual = 0.5 * (left_theta + right_theta)

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

    def get_reference_trajectory(self):
        """Find the nearest path point and build a distance-sampled horizon."""
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
        
        current_v = self.current_state[3]
        target_step_dist = max(current_v * DT, 0.8)
        # Preview the target forward so the nonholonomic vehicle tracks a feasible point
        # Boost lookahead when laterally far to avoid lateral-only targets (car can't strafe)
        lookahead_raw = PREVIEW_BASE + PREVIEW_GAIN * current_v + PREVIEW_CTE_GAIN * cte
        lookahead_dist = np.clip(lookahead_raw, PREVIEW_MIN, PREVIEW_MAX)
        start_idx = self.index_ahead_by_distance(nearest_idx, lookahead_dist)
        last_idx = start_idx
        
        for i in range(HORIZON):
            # 1. Start with the previewed point, then march forward
            if i == 0:
                idx = start_idx
            else:
                idx = self.index_ahead_by_distance(last_idx, target_step_dist)

            row = self.full_path[idx]
            
            # Extract [x, y, yaw, v]
            ref_x = row[COL_X]
            ref_y = row[COL_Y]
            ref_yaw = self.wrap_angle(float(row[COL_YAW]))
            ref_v = float(np.clip(row[COL_V], MIN_REF_V, MAX_REF_V))
            # print(f"Ref Point {i}: ({ref_x:.2f}, {ref_y:.2f}), Yaw: {ref_yaw:.2f}, V: {ref_v:.2f}")
            
            ref_traj.append([ref_x, ref_y, ref_yaw, ref_v])
            last_idx = idx

        # print(f"nearest_idx: {nearest_idx}, ref: {ref_traj[0]}, CTE: {cte:.2f}m")

        # Return shape: (4, HORIZON)
        return np.array(ref_traj).T

    def run(self):
        if not self.initialized:
            rospy.logerr("NeuralMPC failed to initialize; aborting run().")
            return
        # Warm start memory
        self.prev_u_opt = None
        self.prev_x_opt = None
        
        last_u = [0.0, 0.0] # Store last control for fallback

        
        while not rospy.is_shutdown():
            if self.current_state is None:
                print("Current state is None; skipping iteration.")
                self.rate.sleep()
                continue
                
            try:
                # start_time = time.time()
                ref_traj = self.get_reference_trajectory()
                # end_time = time.time()
                # print(f"Reference trajectory generation time: {end_time - start_time:.2f} seconds")
                if ref_traj is None: 
                    print("Reference trajectory is None; skipping iteration.")
                    continue

                # Only try a warm start after we have a previous solution.
                use_warm = (
                    self.prev_u_opt is not None
                    and self.prev_x_opt is not None
                    and np.all(np.isfinite(self.prev_u_opt))
                    and np.all(np.isfinite(self.prev_x_opt))
                )
                # start_time = time.time()
                sol = self.solve_with_guess(ref_traj, use_prev=use_warm)
                # end_time = time.time()
                # print(f"Solver time: {end_time - start_time:.2f} seconds")
                if sol is None:
                    # Either warm start failed or we skipped it; always try a cold start next.
                    sol = self.solve_with_guess(ref_traj, use_prev=False)
                    if sol is None:
                        raise RuntimeError("Solver failed after warm and cold starts")
                
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
                    debug_u = np.array(self.opti.debug.value(self.U_var))
                    debug_x = np.array(self.opti.debug.value(self.X_var))
                    self.prev_u_opt = debug_u
                    self.prev_x_opt = debug_x
                    cmd_v = float(debug_u[0, 0])
                    cmd_s = float(debug_u[1, 0])
                except Exception:
                    # No usable debug info; drop warm-start memory to force cold next loop
                    self.prev_u_opt = None
                    self.prev_x_opt = None
                    cmd_v, cmd_s = last_u # Absolute fallback
                
                msg = AckermannDrive()
                msg.speed = float(cmd_v)
                msg.steering_angle = float(cmd_s)
                self.pub.publish(msg)
            
            # Visualize
            if PUBLISH_DEBUG_MARKERS:
                try:
                    self.publish_markers(ref_traj, self.prev_x_opt)
                except Exception as e:
                    print(f"Viz Error: {e}")

            self.rate.sleep()

    def publish_markers(self, ref_traj, pred_traj):
        """Publishes Reference (Green) and Predicted (Blue) paths to RViz in LOCAL frame"""
        if self.current_state is None: return

        height = -0.5

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
        marker_stamp = rospy.Time(0)  # Use latest TF to avoid RViz flicker when frames lag

        # 1. Reference Path Marker
        marker_ref = Marker()
        marker_ref.header.frame_id = "base_link" # <--- CHANGED: Local frame
        marker_ref.header.stamp = marker_stamp
        marker_ref.ns = "mpc_ref"
        marker_ref.id = 0
        marker_ref.type = Marker.POINTS
        marker_ref.action = Marker.ADD
        marker_ref.pose.orientation.w = 1.0 # <--- Fix: Initialize quaternion
        marker_ref.frame_locked = True
        marker_ref.scale.x = 0.15  # Point width
        marker_ref.scale.y = 0.15  # Point height
        marker_ref.color.a = 1.0
        marker_ref.color.g = 1.0 # Green
        
        for i in range(ref_traj.shape[1]):
            lx, ly = global_to_local(ref_traj[0, i], ref_traj[1, i])
            p = Point()
            p.x = lx
            p.y = ly
            p.z = height
            marker_ref.points.append(p)
        marker_array.markers.append(marker_ref)

        # Heading arrows for reference path
        marker_ref_heading = Marker()
        marker_ref_heading.header.frame_id = "base_link"
        marker_ref_heading.header.stamp = marker_stamp
        marker_ref_heading.ns = "mpc_ref_heading"
        marker_ref_heading.id = 2
        marker_ref_heading.type = Marker.LINE_LIST
        marker_ref_heading.action = Marker.ADD
        marker_ref_heading.pose.orientation.w = 1.0
        marker_ref_heading.frame_locked = True
        marker_ref_heading.scale.x = 0.05  # line width
        marker_ref_heading.color.a = 1.0
        marker_ref_heading.color.g = 0.8
        heading_len = 0.6
        for i in range(ref_traj.shape[1]):
            yaw = self.wrap_angle(ref_traj[2, i] - curr_yaw)  # rotate heading into local frame
            lx, ly = global_to_local(ref_traj[0, i], ref_traj[1, i])
            hx = lx + heading_len * np.cos(yaw)
            hy = ly + heading_len * np.sin(yaw)
            start = Point(x=float(lx), y=float(ly), z=height)
            end = Point(x=float(hx), y=float(hy), z=height)
            marker_ref_heading.points.extend([start, end])
        marker_array.markers.append(marker_ref_heading)

        # 2. Predicted Path Marker
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
            p.z = -0.5
            marker_pred.points.append(p)
        
        marker_array.markers.append(marker_pred)

        # Heading arrows for predicted path
        marker_pred_heading = Marker()
        marker_pred_heading.header.frame_id = "base_link"
        marker_pred_heading.header.stamp = marker_stamp
        marker_pred_heading.ns = "mpc_pred_heading"
        marker_pred_heading.id = 3
        marker_pred_heading.type = Marker.LINE_LIST
        marker_pred_heading.action = Marker.ADD
        marker_pred_heading.pose.orientation.w = 1.0
        marker_pred_heading.frame_locked = True
        marker_pred_heading.scale.x = 0.05
        marker_pred_heading.color.a = 1.0
        marker_pred_heading.color.r = 0.8
        heading_len_pred = 0.6
        for i in range(pred_traj.shape[1]):
            if np.isnan(pred_traj[2, i]):
                continue
            yaw = float(self.wrap_angle(pred_traj[2, i] - curr_yaw))  # local heading
            lx, ly = global_to_local(pred_traj[0, i], pred_traj[1, i])
            if np.isnan(lx) or np.isnan(ly):
                continue
            hx = lx + heading_len_pred * np.cos(yaw)
            hy = ly + heading_len_pred * np.sin(yaw)
            start = Point(x=float(lx), y=float(ly), z=height)
            end = Point(x=float(hx), y=float(hy), z=height)
            marker_pred_heading.points.extend([start, end])

        marker_array.markers.append(marker_pred_heading)

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
