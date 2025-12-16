#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import pandas as pd
import torch
import casadi as ca
import pickle
from scipy.interpolate import splprep, splev
import time
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, get_tera

from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# ============================
# 1. CONFIGURATION
# ============================
MODEL_PATH = 'gem_dynamics.pth'
SCALER_PATH = 'gem_scaler.pkl'
SCALER_ARRAY_PATH = 'gem_scaler_arrays.npz'
PATH_CSV = 'wps.csv'

if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core

HORIZON = 20          
DT = 0.05               
MIN_REF_V = 0.5        
MAX_REF_V = 3.0         
MAX_ACCEL = 2.0        

MAX_STEER_RATE = 0.5
PREVIEW_MIN = 0.0       
PREVIEW_BASE = 0.0      
PREVIEW_GAIN = 0.0      
PREVIEW_MAX = 0.0      

STEER_GAIN_COMP = 1.0
STEER_OFFSET = 0.0

COL_X = 0
COL_Y = 1
COL_YAW = 2
COL_V = 4

class NeuralMPC:
    def __init__(self):
        rospy.init_node('neural_mpc_acados')
        
        self.current_state = None 
        self.last_cmd_steer = 0.0
        self.last_pub_steer = 0.0
        
        self.cte_history = []
        self.rate = rospy.Rate(1.0 / DT)
        self.initialized = False
        self.prev_u_opt = None
        self.prev_x_opt = None
        
        self.log_data = [] # Time, CTE, SteerCmd, SteerAct, YawRate
        self.traj_log = [] # Stores full trajectories for debugging
        
        self._steer_data = 0.0
        self._left_steer = 0.0
        self._right_steer = 0.0
        self._yaw_rate_data = 0.0
        self._left_idx = None
        self._right_idx = None

        if not self.load_scalers(): return

        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        self.weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
        print("Weights loaded.")

        # LOAD PATH
        try:
            df = pd.read_csv(PATH_CSV, header=None)
            raw_path = df.values
            # Clean raw path (Remove duplicates for Spline)
            diff = np.diff(raw_path[:, :2], axis=0)
            dist = np.linalg.norm(diff, axis=1)
            # Keep first point + points that are far enough from previous
            mask = np.concatenate(([True], dist > 1e-4))
            self.full_path = raw_path[mask]
            
            # --- SPLINE FITTING ---
            self.setup_spline()
            
            # --- DYNAMIC VELOCITY PROFILING (Legacy/Fallback) ---
            self.calculate_velocity_profile(self.full_path)
            
            # Recalculate s
            if len(self.full_path) > 1:
                diffs = np.diff(self.full_path[:, [COL_X, COL_Y]], axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                self.path_s = np.concatenate(([0.0], np.cumsum(seg_lens)))
                self.total_path_length = self.path_s[-1]
            else:
                self.path_s = np.array([0.0])
                self.total_path_length = 0.0
        except Exception as e:
            rospy.logerr(f"Could not load path CSV: {e}")
            return

        self.setup_mpc()

        self.pub = rospy.Publisher('/gem/ackermann_cmd', AckermannDrive, queue_size=1)
        self.vis_pub = rospy.Publisher('/gem/mpc_debug', MarkerArray, queue_size=1, latch=True)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)
        rospy.Subscriber('/gem/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/gem/imu', Imu, self.imu_callback)
        
        self.imu_data = {'ax':0, 'ay':0, 'az':0, 'wz':0}

        self.teleport_to_start()
        
        # Neutralize steering (3s wait)
        print("Neutralizing steering for 3s...")
        z_msg = AckermannDrive()
        z_msg.speed = 0.0
        z_msg.steering_angle = 0.0
        for _ in range(30):
            self.pub.publish(z_msg)
            rospy.sleep(0.1)
            
        print("Acados MPC Ready.")
        rospy.on_shutdown(self.save_log)
        self.initialized = True
        
    def save_log(self):
        if self.log_data:
            df = pd.DataFrame(self.log_data, columns=['time', 'cte', 'steer_cmd', 'steer_act', 'yaw_rate', 'cmd_speed', 'speed', 'steer_left', 'steer_right', 'ax', 'ay', 'az', 'wz'])
            df.to_csv('mpc_debug.csv', index=False)
            print("Saved mpc_debug.csv")
            
        if self.traj_log:
            try:
                with open('mpc_trajectories.pkl', 'wb') as f:
                    pickle.dump(self.traj_log, f)
                print("Saved mpc_trajectories.pkl")
            except Exception as e:
                print(f"Failed to save trajectories: {e}")

    def setup_spline(self):
        """
        Fits a B-Spline to the path for smooth interpolation and derivative calculation.
        """
        try:
            # Fit directly to downsampled full_path to ensure index alignment
            x = self.full_path[:, COL_X]
            y = self.full_path[:, COL_Y]
            
            # Append Wrap-Around Buffer
            n_overlap = 50
            if len(x) > n_overlap:
                x = np.concatenate((x, x[:n_overlap]))
                y = np.concatenate((y, y[:n_overlap]))
            
            # Fit Spline
            # s=0.5: Small smoothing to remove quantization noise but keep shape
            self.spline_tck, self.spline_u_pts = splprep([x, y], s=2.0, k=3)
            
            # self.spline_u_pts corresponds to the padded path
            # Original indices 0..N map to self.spline_u_pts[0..N]
            
            print("Spline Fit Successful.")
            
        except Exception as e:
            print(f"Spline fit failed: {e}")
            self.spline_tck = None

    def calculate_velocity_profile(self, path):
        """
        Updates the velocity column (COL_V) based on path curvature (LS Circle Fit 5-Point).
        v_cmd = sqrt(a_lat / k)
        """
        # 1. Least Squares Circle Fit (5-Point Window for Noise Reduction)
        k = np.zeros(len(path))
        window = 5 # Use 5 points centered on i
        half = window // 2
        
        for i in range(half, len(path) - half):
            # Extract local points
            pts = path[i-half : i+half+1, :2] # Shape (5, 2)
            
            # Center points to reduce numerical errors
            centroid = np.mean(pts, axis=0)
            pts_c = pts - centroid
            x = pts_c[:, 0]
            y = pts_c[:, 1]
            
            # Formulate Least Squares: x^2 + y^2 = A*x + B*y + C
            M = np.column_stack((x, y, np.ones(len(x))))
            D = x**2 + y**2
            
            try:
                # Solve linear system
                params, _, _, _ = np.linalg.lstsq(M, D, rcond=None)
                A, B, C = params
                
                xc = A / 2.0
                yc = B / 2.0
                R_sq = C + xc**2 + yc**2
                R = np.sqrt(max(R_sq, 1e-6))
                
                # Curvature k = 1/R
                k[i] = 1.0 / (R + 1e-6)
            except np.linalg.LinAlgError:
                k[i] = 0.0
                
        # Pad endpoints
        k[:half] = k[half]
        k[-half:] = k[-half-1]
        
        # 2. Compute Velocity Limit
        max_lat_accel = 0.4 # m/s^2 (Tuned for 20m radius turns)
        
        for i in range(len(path)):
            curvature = k[i]
            # v^2/r = a_lat => v = sqrt(a_lat * r) = sqrt(a_lat / k)
            v_limit = np.sqrt(max_lat_accel / (curvature + 1e-3))
            
            # Apply Limit (Smoothly)
            path[i, COL_V] = min(MAX_REF_V, v_limit)
            
            # Enforce Min Speed
            path[i, COL_V] = max(path[i, COL_V], MIN_REF_V)
            
        print("Velocity Profile Generated based on Curvature (Menger).")


    def load_scalers(self):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scalers = pickle.load(f)
                self.sx_mean = np.asarray(scalers['x'].mean_, dtype=float)
                self.sx_scale = np.asarray(scalers['x'].scale_, dtype=float)
                self.sy_mean = np.asarray(scalers['y'].mean_, dtype=float)
                self.sy_scale = np.asarray(scalers['y'].scale_, dtype=float)
            return True
        except Exception:
            try:
                arrs = np.load(SCALER_ARRAY_PATH)
                self.sx_mean = arrs['x_mean']
                self.sx_scale = arrs['x_scale']
                self.sy_mean = arrs['y_mean']
                self.sy_scale = arrs['y_scale']
                return True
            except Exception: return False

    def create_acados_model(self):
        model = AcadosModel()
        
        x = ca.SX.sym('x', 7)
        u = ca.SX.sym('u', 2)
        p = ca.SX.sym('p', 4)

        v_act = x[3]
        steer_act = x[4]
        yaw_rate = x[5]
        
        cmd_v = u[0]
        cmd_s = u[1]

        nn_input = ca.vertcat(v_act, steer_act, yaw_rate, cmd_v, cmd_s, DT)
        
        sx_mean = ca.DM(self.sx_mean)
        sx_scale = ca.DM(self.sx_scale)
        sy_mean = ca.DM(self.sy_mean)
        sy_scale = ca.DM(self.sy_scale)
        
        inp_norm = (nn_input - sx_mean) / sx_scale
        
        act = ca.tanh
        w1 = ca.DM(self.weights['net.0.weight']); b1 = ca.DM(self.weights['net.0.bias'])
        h1 = act(ca.mtimes(w1, inp_norm) + b1)
        w2 = ca.DM(self.weights['net.2.weight']); b2 = ca.DM(self.weights['net.2.bias'])
        h2 = act(ca.mtimes(w2, h1) + b2)
        w3 = ca.DM(self.weights['net.4.weight']); b3 = ca.DM(self.weights['net.4.bias'])
        h3 = act(ca.mtimes(w3, h2) + b3)
        w4 = ca.DM(self.weights['net.6.weight']); b4 = ca.DM(self.weights['net.6.bias'])
        out_norm = ca.mtimes(w4, h3) + b4
        
        out_real = out_norm * sy_scale + sy_mean
        
        # Dynamics (No HACKS needed with augmented training)
        d_yaw = out_real[2] 
        yaw_curr = x[2]
        
        dx_g = ca.cos(yaw_curr)*out_real[0] - ca.sin(yaw_curr)*out_real[1]
        dy_g = ca.sin(yaw_curr)*out_real[0] + ca.cos(yaw_curr)*out_real[1]
        d_v = out_real[3]
        d_steer = out_real[4]
        next_yaw_rate = d_yaw / DT 

        x_next = ca.vertcat(
            x[0] + dx_g, x[1] + dy_g, x[2] + d_yaw, x[3] + d_v, x[4] + d_steer,
            next_yaw_rate, cmd_s
        )

        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = x_next
        model.name = "gem_nn_mpc"
        return model

    def setup_mpc(self):
        if AcadosOcp is None: return
        
        if get_tera is not None:
            try: get_tera(tera_version="0.0.34")
            except: pass

        model = self.create_acados_model()
        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(4) 
        
        if hasattr(ocp.solver_options, "N_horizon"):
            ocp.solver_options.N_horizon = HORIZON
        else:
            ocp.dims.N = HORIZON
            
        nx = 7; nu = 2
        self.nx = nx; self.nu = nu

        # --- TUNED COST FUNCTION FOR STABILITY ---
        Q_lat = 1.0 
        Q_yaw = 1.0     
        Q_vel = 5.0     
        
        R_accel = 1.0   
        R_steer = 20.0   
        
        R_rate  = 2000.0 
        
        Q_steer_state = 0.0 
        
        x_sym = model.x # [x, y, yaw, v, steer, yaw_rate, cmd_s]
        u_sym = model.u # [cmd_v, cmd_s]
        p_sym = model.p # [p_x, p_y, p_yaw, p_v]
        
        # Errors
        e_pos = (x_sym[0] - p_sym[0])**2 + (x_sym[1] - p_sym[1])**2
        yaw_diff = x_sym[2] - p_sym[2]
        e_yaw = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))
        e_v = x_sym[3] - p_sym[3]
        diff_steer = u_sym[1] - x_sym[6]
        diff_v = u_sym[0] - x_sym[3]

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        
        y_expr = ca.vertcat(e_pos, e_yaw, e_v, diff_v, u_sym[1], x_sym[4], diff_steer)
        ocp.model.cost_y_expr = y_expr
        
        # Weights
        W = np.diag([Q_lat, Q_yaw, Q_vel, R_accel, R_steer, Q_steer_state, R_rate])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(7)

        y_expr_e = ca.vertcat(e_pos, e_yaw, e_v, x_sym[4])
        ocp.model.cost_y_expr_e = y_expr_e
        W_e = np.diag([Q_lat, Q_yaw, Q_vel, Q_steer_state])
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = np.zeros(4)

        # --- Constraints ---
        ocp.constraints.idxbx = np.array([3, 4]) 
        ocp.constraints.lbx = np.array([0.0, -0.65])
        ocp.constraints.ubx = np.array([MAX_REF_V, 0.65])
        
        # --- Soft Constraints ---
        # Add slack for state bounds (idxbx)
        ocp.constraints.idxsbx = np.array([0, 1]) # Corresponds to indices in idxbx (so index 3 and 4 of x)
        ns = 2
        ocp.cost.zl = 1000 * np.ones((ns,)) # Linear penalty (lower)
        ocp.cost.zu = 1000 * np.ones((ns,)) # Linear penalty (upper)
        ocp.cost.Zl = 100 * np.ones((ns,))  # Quadratic penalty (lower)
        ocp.cost.Zu = 100 * np.ones((ns,))  # Quadratic penalty (upper)
        
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([0.0, -0.6])
        ocp.constraints.ubu = np.array([MAX_REF_V, 0.6])

        ocp.constraints.x0 = np.zeros(nx)

        # --- HARD CONSTRAINT: STEERING RATE ---
        # limit (u_steer - x_prev_steer) / DT <= MAX_RATE
        # => -MAX_RATE*DT <= u[1] - x[6] <= MAX_RATE*DT
        
        # C*x + D*u
        # C = [0, 0, 0, 0, 0, 0, -1]
        # D = [0, 1]
        
        ocp.constraints.C = np.zeros((1, nx))
        ocp.constraints.C[0, 6] = -1.0
        
        ocp.constraints.D = np.zeros((1, nu))
        ocp.constraints.D[0, 1] = 1.0
        
        d_max = MAX_STEER_RATE * DT
        ocp.constraints.lg = np.array([-d_max])
        ocp.constraints.ug = np.array([ d_max])

        # --- Solver ---
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.tf = HORIZON * DT
        ocp.solver_options.print_level = 0
        
        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    def build_initial_guess(self, ref_traj, use_prev=True):
        curr_phys = self.current_state 
        curr_full = np.append(curr_phys, self.last_cmd_steer)
        
        if use_prev and self.prev_u_opt is not None:
            u_guess = np.hstack((self.prev_u_opt[:, 1:], self.prev_u_opt[:, -1:]))
        else:
            u_guess = np.zeros((2, HORIZON))
            u_guess[0, :] = curr_phys[3]
            tgt = ref_traj[:, 0]
            heading = np.arctan2(tgt[1]-curr_phys[1], tgt[0]-curr_phys[0])
            err = self.wrap_angle(heading - curr_phys[2])
            u_guess[1, :] = np.clip(err, -0.5, 0.5)

        x_guess = np.zeros((7, HORIZON + 1))
        x_guess[:, 0] = curr_full
        
        temp_state = curr_full.copy()
        
        for k in range(HORIZON):
            v, st, yr = temp_state[3], temp_state[4], temp_state[5]
            cv, cs = u_guess[0,k], u_guess[1,k]
            
            inp = np.array([v, st, yr, cv, cs, DT])
            inp_n = (inp - self.sx_mean)/self.sx_scale
            
            act = np.tanh
            h1 = act(self.weights['net.0.weight'].dot(inp_n) + self.weights['net.0.bias'])
            h2 = act(self.weights['net.2.weight'].dot(h1) + self.weights['net.2.bias'])
            h3 = act(self.weights['net.4.weight'].dot(h2) + self.weights['net.4.bias'])
            out_n = self.weights['net.6.weight'].dot(h3) + self.weights['net.6.bias']
            out_r = out_n * self.sy_scale + self.sy_mean
            
            d_yaw = out_r[2]
            yaw = temp_state[2]
            dx = np.cos(yaw)*out_r[0] - np.sin(yaw)*out_r[1]
            dy = np.sin(yaw)*out_r[0] + np.cos(yaw)*out_r[1]
            dv = out_r[3]
            dsteer = out_r[4]
            
            next_st = np.zeros(7)
            next_st[0] = temp_state[0] + dx
            next_st[1] = temp_state[1] + dy
            next_st[2] = self.wrap_angle(temp_state[2] + d_yaw)
            next_st[3] = temp_state[3] + dv
            next_st[4] = temp_state[4] + dsteer
            next_st[5] = d_yaw / DT
            next_st[6] = cs
            
            x_guess[:, k+1] = next_st
            temp_state = next_st
            
        return x_guess, u_guess

    def compensate_latency(self, x_curr):
        """
        Projects the state forward by DT (50ms) to account for Solver + Actuator Latency.
        Uses a simple Kinematic Bicycle Model.
        """
        dt = 0.13 # INCREASED: Compensate for ~130ms latency (Solver + ROS + Actuator)
        x, y, yaw, v, steer, yaw_rate = x_curr
        
        # Kinematic Projection
        # We assume velocity and steering angle don't change instantly
        nx = x + v * np.cos(yaw) * dt
        ny = y + v * np.sin(yaw) * dt
        nyaw = self.wrap_angle(yaw + yaw_rate * dt)
        
        # Simple damping for dynamics states
        nv = v 
        nsteer = steer 
        nyaw_rate = yaw_rate 
        
        return np.array([nx, ny, nyaw, nv, nsteer, nyaw_rate])

    def solve_with_guess(self, ref_traj, use_prev=True):
        try:
            x_guess, u_guess = self.build_initial_guess(ref_traj, use_prev)
            
            # --- APPLY LATENCY COMPENSATION ---
            # Instead of using current_state directly, we use the predicted future state
            x0_lat = self.compensate_latency(self.current_state)
            
            # Append last command (standard procedure)
            x0 = np.append(x0_lat, self.last_cmd_steer)
            
            self.solver.set(0, 'lbx', x0)
            self.solver.set(0, 'ubx', x0)
            
            # ... (The rest of the solve loop remains identical) ...
            for k in range(HORIZON):
                self.solver.set(k, 'yref', np.zeros(7))
                self.solver.set(k, 'p', ref_traj[:4, k])
                self.solver.set(k, 'x', x_guess[:, k])
                self.solver.set(k, 'u', u_guess[:, k])
                
            self.solver.set(HORIZON, 'p', ref_traj[:4, -1])
            self.solver.set(HORIZON, 'x', x_guess[:, -1])
            
            status = self.solver.solve()
            if status != 0: return None
            
            u_opt = np.zeros((2, HORIZON))
            x_opt = np.zeros((7, HORIZON + 1))
            for k in range(HORIZON):
                u_opt[:, k] = self.solver.get(k, 'u')
                x_opt[:, k] = self.solver.get(k, 'x')
            x_opt[:, HORIZON] = self.solver.get(HORIZON, 'x')
            
            return x_opt, u_opt
            
        except Exception:
            if use_prev: 
                self.prev_u_opt = None
                self.prev_x_opt = None
            return None
            
        except Exception:
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
            
            self.current_state = np.array([
                p.x, p.y, yaw, speed, self._steer_data, w_ang.z
            ])
            self._yaw_rate_data = w_ang.z
            
        except ValueError: pass

    def joint_state_callback(self, msg):
        try:
            if self._left_idx is None:
                self._left_idx = msg.name.index("left_steering_hinge_joint")
                self._right_idx = msg.name.index("right_steering_hinge_joint")
            l = msg.position[self._left_idx]
            r = msg.position[self._right_idx]
            self._left_steer = l
            self._right_steer = r
            self._steer_data = (l + r) / 2.0
        except ValueError: pass
            
    def imu_callback(self, msg):
        self.imu_data['ax'] = msg.linear_acceleration.x
        self.imu_data['ay'] = msg.linear_acceleration.y
        self.imu_data['az'] = msg.linear_acceleration.z
        self.imu_data['wz'] = msg.angular_velocity.z

    @staticmethod
    def wrap_angle(angle):
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle

    def index_ahead_by_distance(self, start_idx, dist):
        if len(self.path_s) == 0: return 0
        t_s = (self.path_s[start_idx] + dist) % self.total_path_length
        idx = np.searchsorted(self.path_s, t_s)
        return min(idx, len(self.path_s)-1)

    def calc_closest_index(self, path, state):
        dx = path[:, COL_X] - state[0]
        dy = path[:, COL_Y] - state[1]
        d = np.hypot(dx, dy)
        min_idx = np.argmin(d)
        min_dist = d[min_idx]
        return min_idx, min_dist

    def get_reference_trajectory(self):
        if self.current_state is None: return None
        cx, cy = self.current_state[0], self.current_state[1]
        
        # 1. Start at Nearest Point (Corrects "Ghost Error")
        dists = np.linalg.norm(self.full_path[:, [COL_X, COL_Y]] - np.array([cx, cy]), axis=1)
        nearest_idx = np.argmin(dists)
        
        # SIGNED CTE Calculation
        # Vector from path to car
        dx_err = cx - self.full_path[nearest_idx, COL_X]
        dy_err = cy - self.full_path[nearest_idx, COL_Y]
        
        # Path heading
        path_yaw = self.full_path[nearest_idx, COL_YAW]
        
        # Cross product (z-component) to find which side: (cos*dy - sin*dx)
        # If > 0, left of path. If < 0, right of path.
        cte_signed = np.sin(path_yaw) * dx_err - np.cos(path_yaw) * dy_err
        # Check sign convention vs standard Frenet frame
        # Standard: Left is positive.
        # Tangent = [cos, sin]. Normal = [-sin, cos].
        # Error projected on Normal = (dx, dy) . (-sin, cos) = -sin*dx + cos*dy
        cte_signed = -np.sin(path_yaw)*dx_err + np.cos(path_yaw)*dy_err
        
        self.cte_history.append(cte_signed)
        
        v = max(self.current_state[3], 0.1)
        step_dist = v * DT
        
        ref = []
        curr = nearest_idx
        for _ in range(HORIZON):
            row = self.full_path[curr]
        # 1. Find Closest Point
        min_idx, min_dist = self.calc_closest_index(self.full_path, self.current_state)
        
        # 2. Setup Spline References
        traj = np.zeros((HORIZON, 7))
        dt = 0.05 # Prediction Step
        
        if self.spline_tck is None:
            # Fallback to old discrete logic if spline messed up
            idx = min_idx
            for i in range(HORIZON):
                goal_pt = self.full_path[idx]
                traj[i, 0] = goal_pt[COL_X]
                traj[i, 1] = goal_pt[COL_Y]
                traj[i, 2] = goal_pt[COL_YAW]
                traj[i, 3] = goal_pt[COL_V]
                
                # Advance
                dist_step = max(0.2, self.current_state[3] * dt) # dynamic step
                idx = self.index_ahead_by_distance(idx, dist_step)
        else:
            # --- SPLINE BASED SAMPLING ---
            
            # Get starting u
            # Mapping: idx -> spline_u_pts[idx]
            # Since full_path indices align with first part of spline_u_pts
            current_u = self.spline_u_pts[min_idx]
            
            # Refine u (Find local minimum distance on spline) - Optional, but good accuracy
            # For now, just use the waypoint u.
            
            tck = self.spline_tck
            
            for i in range(HORIZON):
                # Eval Position
                x_ref, y_ref = splev(current_u, tck)
                
                # Eval 1st Deriv (Tangent) -> Yaw & Arc Length scaling
                dx, dy = splev(current_u, tck, der=1)
                norm_deriv = np.hypot(dx, dy)
                yaw_ref = np.arctan2(dy, dx)
                
                # Eval 2nd Deriv -> Curvature
                ddx, ddy = splev(current_u, tck, der=2)
                curvature = (dx * ddy - dy * ddx) / (norm_deriv**3 + 1e-6)
                
                # Dynamic Velocity Limit (Analytic Curvature)
                max_lat_accel = 0.4 # Tuned
                v_limit = np.sqrt(max_lat_accel / (np.abs(curvature) + 1e-3))
                v_ref = min(MAX_REF_V, v_limit)
                v_ref = max(v_ref, MIN_REF_V)
                
                # Fill State (x, y, yaw, v, steer, yawrate, cmd_s)
                traj[i, 0] = x_ref
                traj[i, 1] = y_ref
                traj[i, 2] = yaw_ref
                traj[i, 3] = v_ref
                
                # Advance u
                # du = ds / (ds/du) = (v * dt) / norm_deriv
                ds = v_ref * dt
                du = ds / (norm_deriv + 1e-6)
                current_u += du
                
                # Check Bounds (No Loop Wrapping needed if buffer is large enough)
                # If buffer is 50pts approx 50m, and horizon is 20 steps * 3m/s * 0.05s = 3m. 
                # Buffer is plenty.
                if current_u > self.spline_u_pts[-1]:
                    current_u = self.spline_u_pts[-1]
                    
        return traj.T

    def run(self):
        if not self.initialized: return
        
        while not rospy.is_shutdown():
            if self.current_state is None:
                self.rate.sleep()
                continue
                
            ref_traj = self.get_reference_trajectory()
            use_warm = (self.prev_u_opt is not None)
            
            res = self.solve_with_guess(ref_traj, use_warm)
            if res is None and use_warm:
                res = self.solve_with_guess(ref_traj, False)
                
            msg = AckermannDrive()
            if res is not None:
                x_opt, u_opt = res
                self.prev_u_opt = u_opt
                self.prev_x_opt = x_opt
                
                cmd_v = float(u_opt[0, 0])
                cmd_s = float(u_opt[1, 0])
                
                self.last_cmd_steer = cmd_s 
                
                # --- OUTPUT FILTER ---
                # Smooth the command to prevent high-frequency oscillation
                ALPHA = 0.9
                filtered_cmd = ALPHA * cmd_s + (1 - ALPHA) * self.last_pub_steer
                self.last_pub_steer = filtered_cmd
                
                final_pub_steer = (filtered_cmd / STEER_GAIN_COMP) + STEER_OFFSET
                
                msg.speed = cmd_v
                msg.steering_angle = final_pub_steer
                
                if self.vis_pub.get_num_connections() > 0:
                    self.publish_markers(ref_traj, x_opt)
            else:
                msg.speed = 0.0
                print("Acados Solver Failed -> Stopping")

            self.pub.publish(msg)
            
            # LOGGING
            if self.current_state is not None:
                t = rospy.get_time()
                # Time, CTE, SteerCmd, SteerActual, YawRate, Speed
                try:
                    cte = self.cte_history[-1] if self.cte_history else 0.0
                    row = [t, cte, msg.steering_angle, self._steer_data, self._yaw_rate_data, msg.speed, self.current_state[3], self._left_steer, self._right_steer,
                           self.imu_data['ax'], self.imu_data['ay'], self.imu_data['az'], self.imu_data['wz']]
                    self.log_data.append(row)
                    
                    # Log trajectories (Ref and Pred)
                    if res is not None:
                         # Store as simple arrays to save space in pickle
                         self.traj_log.append({
                             'time': t,
                             'ref': ref_traj,
                             'pred': res[0] # x_opt
                         })
                except: pass

            self.rate.sleep()
            
            self.rate.sleep()

    def publish_markers(self, ref_traj, x_opt):
        if self.current_state is None: return
        cx, cy, cyaw = self.current_state[:3]
        cos_y, sin_y = np.cos(cyaw), np.sin(cyaw)
        
        def g2l(gx, gy):
            dx, dy = gx-cx, gy-cy
            return (cos_y*dx + sin_y*dy), (-sin_y*dx + cos_y*dy)
            
        ma = MarkerArray()
        stamp = rospy.Time(0)
        
        m1 = Marker()
        m1.header.frame_id = "base_link"
        m1.header.stamp = stamp
        m1.ns = "mpc_ref"
        m1.id = 0
        m1.type = Marker.POINTS
        m1.action = Marker.ADD
        m1.pose.orientation.w = 1.0
        m1.scale.x = 0.15; m1.scale.y = 0.15
        m1.color.a = 1.0; m1.color.g = 1.0
        for i in range(ref_traj.shape[1]):
            lx, ly = g2l(ref_traj[0,i], ref_traj[1,i])
            p = Point(x=lx, y=ly, z=-0.5)
            m1.points.append(p)
        ma.markers.append(m1)
        
        if x_opt is not None:
            m2 = Marker()
            m2.header.frame_id = "base_link"
            m2.header.stamp = stamp
            m2.ns = "mpc_pred"
            m2.id = 1
            m2.type = Marker.SPHERE_LIST
            m2.action = Marker.ADD
            m2.pose.orientation.w = 1.0
            m2.scale.x=0.1; m2.scale.y=0.1; m2.scale.z=0.1
            m2.color.a=1.0; m2.color.b=1.0
            for i in range(x_opt.shape[1]):
                lx, ly = g2l(x_opt[0,i], x_opt[1,i])
                if np.isnan(lx): continue
                p = Point(x=lx, y=ly, z=-0.5)
                m2.points.append(p)
            ma.markers.append(m2)
            
        self.vis_pub.publish(ma)

    def teleport_to_start(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            sx = self.full_path[0, COL_X]
            sy = self.full_path[0, COL_Y]
            syaw = float(self.full_path[0, COL_YAW])
            
            s = ModelState()
            s.model_name = 'gem'
            s.pose.position.x = sx
            s.pose.position.y = sy
            s.pose.position.z = 2.0
            q = quaternion_from_euler(0,0,syaw)
            s.pose.orientation.x = q[0]
            s.pose.orientation.y = q[1]
            s.pose.orientation.z = q[2]
            s.pose.orientation.w = q[3]
            
            rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(s)
            self.pub.publish(AckermannDrive())
        except Exception: pass

if __name__ == '__main__':
    c = NeuralMPC()
    try: c.run()
    except rospy.ROSInterruptException: pass