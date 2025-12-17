#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import sys
import pandas as pd
import torch
import casadi as ca
import pickle
from scipy.interpolate import splprep, splev
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, get_tera

from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ============================
# 1. CONFIGURATION
# ============================
MODEL_PATH = 'gem_dynamics.pth'
SCALER_PATH = 'gem_scaler.pkl'
SCALER_ARRAY_PATH = 'gem_scaler_arrays.npz'
PATH_CSV = 'wps.csv'

DEBUG_CSV_PATH = 'mpc_debug.csv'
TRAJ_PKL_PATH = 'mpc_trajectories.pkl'
RMSE_PLOT_PATH = 'model_rmse.png'

if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core

HORIZON = 20
DT = 0.05

MIN_REF_V = 0.5
MAX_REF_V = 4.5

MAX_STEER_RATE = 0.5

STEER_GAIN_COMP = 1.0
STEER_OFFSET = 0.0

# Robust yaw cost around +/-pi
USE_YAW_SINCOS_COST = True

# Warm start reset on yaw wrap jumps (highly recommended)
RESET_WARMSTART_ON_YAW_WRAP_JUMP = True
YAW_WRAP_JUMP_THRESH = 3.0  # rad

# RMSE plot settings
RMSE_ROLLING_WINDOW_SEC = 5.0  # rolling RMSE window
RMSE_MIN_POINTS = 30           # min points to show RMSE

COL_X = 0
COL_Y = 1
COL_YAW = 2
COL_V = 4


class NeuralMPC:
    def __init__(self):
        rospy.init_node('neural_mpc_acados')

        # current_state: [x, y, yaw_cont, v, steer, yaw_rate]
        self.current_state = None
        self.last_cmd_steer = 0.0
        self.last_pub_steer = 0.0

        self.cte_history = []
        self.rate = rospy.Rate(1.0 / DT)
        self.initialized = False

        self.prev_u_opt = None
        self.prev_x_opt = None

        self.log_data = []
        self.traj_log = []

        self._steer_data = 0.0
        self._left_steer = 0.0
        self._right_steer = 0.0
        self._yaw_rate_data = 0.0
        self._left_idx = None
        self._right_idx = None

        self.imu_data = {'ax': 0, 'ay': 0, 'az': 0, 'wz': 0}

        # --- yaw unwrapping state (fix crab around +/-pi) ---
        self._yaw_meas_prev = None   # wrapped yaw in [-pi, pi]
        self._yaw_cont = None        # continuous yaw (can exceed +/-pi)
        self._yaw_meas_wrapped = 0.0

        if not self.load_scalers():
            rospy.logerr("Failed to load scalers.")
            return

        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        self.weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
        print("Weights loaded.")

        # LOAD PATH
        try:
            df = pd.read_csv(PATH_CSV, header=None)
            raw_path = df.values

            if len(raw_path) > 1:
                diff = np.diff(raw_path[:, :2], axis=0)
                dist = np.linalg.norm(diff, axis=1)
                mask = np.concatenate(([True], dist > 1e-4))
                self.full_path = raw_path[mask]
            else:
                self.full_path = raw_path

            self.setup_spline()
            self.calculate_velocity_profile(self.full_path)

            if len(self.full_path) > 1:
                diffs = np.diff(self.full_path[:, [COL_X, COL_Y]], axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                self.path_s = np.concatenate(([0.0], np.cumsum(seg_lens)))
                self.total_path_length = float(self.path_s[-1])
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

        self.teleport_to_start()

        # Neutralize steering
        print("Neutralizing steering for 3s...")
        z_msg = AckermannDrive()
        z_msg.speed = 0.0
        z_msg.steering_angle = 0.0
        for _ in range(30):
            self.pub.publish(z_msg)
            rospy.sleep(0.1)

        print("Acados MPC Ready.")
        rospy.on_shutdown(self.save_log_and_rmse_plot)
        self.initialized = True

    # ============================
    # Logging + RMSE plot on shutdown
    # ============================
    def save_log_and_rmse_plot(self):
        if not self.log_data:
            print("No log data to save.")
            return

        cols = [
            'time', 'cte',
            'cmd_speed_pub',          # published speed
            'steer_cmd_pub',          # published steering (after filter+gain/offset)
            'steer_cmd_opt',          # raw solver steer (before filter+gain/offset)
            'speed_meas',             # measured speed
            'steer_act',              # measured steering
            'yaw_rate',               # measured yaw rate

            'state_x', 'state_y', 'yaw_cont',
            'yaw_meas_wrapped', 'ref_yaw0',

            'steer_left', 'steer_right',
            'ax', 'ay', 'az', 'wz',
        ]

        df = pd.DataFrame(self.log_data, columns=cols)
        df.to_csv(DEBUG_CSV_PATH, index=False)
        print(f"Saved {DEBUG_CSV_PATH}")

        if self.traj_log:
            try:
                with open(TRAJ_PKL_PATH, 'wb') as f:
                    pickle.dump(self.traj_log, f)
                print(f"Saved {TRAJ_PKL_PATH}")
            except Exception as e:
                print(f"Failed to save trajectories: {e}")

        # RMSE plot (model one-step prediction accuracy)
        try:
            self._plot_model_rmse(df)
        except Exception as e:
            print(f"[WARN] RMSE plot failed: {e}")

    def _nn_predict_real(self, v, steer, yaw_rate, cmd_v, cmd_steer, dt):
        """
        Returns NN outputs in REAL units:
          [dx_local_mid, dy_local_mid, d_yaw, d_v, d_steer]
        """
        inp = np.array([v, steer, yaw_rate, cmd_v, cmd_steer, dt], dtype=float)
        inp_n = (inp - self.sx_mean) / self.sx_scale

        h1 = np.tanh(self.weights['net.0.weight'].dot(inp_n) + self.weights['net.0.bias'])
        h2 = np.tanh(self.weights['net.2.weight'].dot(h1) + self.weights['net.2.bias'])
        h3 = np.tanh(self.weights['net.4.weight'].dot(h2) + self.weights['net.4.bias'])
        out_n = self.weights['net.6.weight'].dot(h3) + self.weights['net.6.bias']

        out_r = out_n * self.sy_scale + self.sy_mean
        return out_r

    def _plot_model_rmse(self, df: pd.DataFrame):
        """
        Compute one-step prediction errors of the NN model versus measured next state,
        then plot rolling RMSE over time.
        """
        t = df['time'].to_numpy(dtype=float)
        if len(t) < RMSE_MIN_POINTS:
            print("[RMSE] Not enough samples to compute RMSE.")
            return

        x = df['state_x'].to_numpy(dtype=float)
        y = df['state_y'].to_numpy(dtype=float)
        yaw = df['yaw_cont'].to_numpy(dtype=float)         # continuous yaw (important)
        v = df['speed_meas'].to_numpy(dtype=float)
        steer = df['steer_act'].to_numpy(dtype=float)
        yaw_rate = df['yaw_rate'].to_numpy(dtype=float)

        cmd_v = df['cmd_speed_pub'].to_numpy(dtype=float)
        cmd_s = df['steer_cmd_pub'].to_numpy(dtype=float)

        dt = np.diff(t)
        # guard weird dt
        dt = np.clip(dt, 1e-3, 0.2)

        n = len(t) - 1
        pos_err = np.zeros(n, dtype=float)
        yaw_err = np.zeros(n, dtype=float)
        v_err = np.zeros(n, dtype=float)
        steer_err = np.zeros(n, dtype=float)

        for i in range(n):
            out = self._nn_predict_real(
                v=v[i],
                steer=steer[i],
                yaw_rate=yaw_rate[i],
                cmd_v=cmd_v[i],
                cmd_steer=cmd_s[i],
                dt=float(dt[i])
            )

            dx_local, dy_local, d_yaw, d_v, d_steer = out
            yaw_mid = yaw[i] + 0.5 * d_yaw

            # Option A consistent global update using midpoint yaw
            dx_g = np.cos(yaw_mid) * dx_local - np.sin(yaw_mid) * dy_local
            dy_g = np.sin(yaw_mid) * dx_local + np.cos(yaw_mid) * dy_local

            x_pred = x[i] + dx_g
            y_pred = y[i] + dy_g
            yaw_pred = yaw[i] + d_yaw
            v_pred = v[i] + d_v
            steer_pred = steer[i] + d_steer

            # Compare to measured next step
            dxp = x_pred - x[i + 1]
            dyp = y_pred - y[i + 1]
            pos_err[i] = np.hypot(dxp, dyp)

            yaw_err[i] = self.wrap_angle((yaw_pred - yaw[i + 1]))
            v_err[i] = (v_pred - v[i + 1])
            steer_err[i] = (steer_pred - steer[i + 1])

        # Global RMSE (scalar per channel)
        rmse_pos = float(np.sqrt(np.mean(pos_err**2)))
        rmse_yaw = float(np.sqrt(np.mean(yaw_err**2)))
        rmse_v = float(np.sqrt(np.mean(v_err**2)))
        rmse_steer = float(np.sqrt(np.mean(steer_err**2)))

        print("[RMSE] One-step model accuracy (global):")
        print(f"  pos_rmse    = {rmse_pos:.6f} m")
        print(f"  yaw_rmse    = {rmse_yaw:.6f} rad")
        print(f"  v_rmse      = {rmse_v:.6f} m/s")
        print(f"  steer_rmse  = {rmse_steer:.6f} rad")

        if plt is None:
            print("[RMSE] matplotlib not available, skipping plot.")
            return

        # Rolling RMSE
        window_n = max(5, int(round(RMSE_ROLLING_WINDOW_SEC / DT)))
        idx_t = t[1:]  # errors aligned to next sample time

        def rolling_rmse(e):
            return np.sqrt(pd.Series(e**2).rolling(window_n, min_periods=max(5, window_n // 5)).mean()).to_numpy()

        r_pos = rolling_rmse(pos_err)
        r_yaw = rolling_rmse(yaw_err)
        r_v = rolling_rmse(v_err)
        r_steer = rolling_rmse(steer_err)

        plt.figure(figsize=(12, 8))
        plt.plot(idx_t, r_pos, label='pos RMSE (m)')
        plt.plot(idx_t, r_yaw, label='yaw RMSE (rad)')
        plt.plot(idx_t, r_v, label='v RMSE (m/s)')
        plt.plot(idx_t, r_steer, label='steer RMSE (rad)')
        plt.grid(True)
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel(f'rolling RMSE (window={RMSE_ROLLING_WINDOW_SEC:.1f}s)')
        plt.title('NN Dynamics Model One-step Prediction RMSE')

        plt.tight_layout()
        plt.savefig(RMSE_PLOT_PATH, dpi=150)
        print(f"[RMSE] Saved plot: {RMSE_PLOT_PATH}")

        # Best-effort "display"
        # In many ROS/Gazebo runs this is headless; saving is the reliable part.
        if os.environ.get("DISPLAY", ""):
            try:
                plt.show(block=False)
                plt.pause(0.1)
            except Exception:
                pass

    # ============================
    # Spline + velocity profile
    # ============================
    def setup_spline(self):
        try:
            x = self.full_path[:, COL_X]
            y = self.full_path[:, COL_Y]

            n_overlap = 50
            if len(x) > n_overlap:
                x = np.concatenate((x, x[:n_overlap]))
                y = np.concatenate((y, y[:n_overlap]))

            self.spline_tck, self.spline_u_pts = splprep([x, y], s=2.0, k=3)
            print("Spline Fit Successful.")
        except Exception as e:
            print(f"Spline fit failed: {e}")
            self.spline_tck = None
            self.spline_u_pts = None

    def calculate_velocity_profile(self, path):
        if path is None or len(path) < 5:
            return

        k = np.zeros(len(path))
        window = 5
        half = window // 2

        for i in range(half, len(path) - half):
            pts = path[i - half: i + half + 1, :2]
            centroid = np.mean(pts, axis=0)
            pts_c = pts - centroid
            xx = pts_c[:, 0]
            yy = pts_c[:, 1]

            M = np.column_stack((xx, yy, np.ones(len(xx))))
            D = xx**2 + yy**2

            try:
                params, _, _, _ = np.linalg.lstsq(M, D, rcond=None)
                A, B, C = params
                xc = A / 2.0
                yc = B / 2.0
                R_sq = C + xc**2 + yc**2
                R = np.sqrt(max(R_sq, 1e-6))
                k[i] = 1.0 / (R + 1e-6)
            except np.linalg.LinAlgError:
                k[i] = 0.0

        k[:half] = k[half]
        k[-half:] = k[-half - 1]

        max_lat_accel = 0.4
        for i in range(len(path)):
            curvature = k[i]
            v_limit = np.sqrt(max_lat_accel / (curvature + 1e-3))
            path[i, COL_V] = min(MAX_REF_V, float(v_limit))
            path[i, COL_V] = max(path[i, COL_V], MIN_REF_V)

        print("Velocity Profile Generated based on Curvature.")

    # ============================
    # Scalers
    # ============================
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
            except Exception:
                return False

    # ============================
    # Option A: midpoint yaw rotation in model
    # ============================
    def create_acados_model(self):
        model = AcadosModel()

        # x = [x, y, yaw_cont, v, steer, yaw_rate, last_cmd_steer]
        x = ca.SX.sym('x', 7)
        u = ca.SX.sym('u', 2)  # [cmd_v, cmd_s]
        p = ca.SX.sym('p', 4)  # [ref_x, ref_y, ref_yaw_cont, ref_v]

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
        # out_real = [dx_local_mid, dy_local_mid, d_yaw, d_v, d_steer]

        d_yaw = out_real[2]
        yaw_curr = x[2]
        yaw_mid = yaw_curr + 0.5 * d_yaw

        dx_g = ca.cos(yaw_mid) * out_real[0] - ca.sin(yaw_mid) * out_real[1]
        dy_g = ca.sin(yaw_mid) * out_real[0] + ca.cos(yaw_mid) * out_real[1]

        d_v = out_real[3]
        d_steer = out_real[4]
        next_yaw_rate = d_yaw / DT

        x_next = ca.vertcat(
            x[0] + dx_g,
            x[1] + dy_g,
            x[2] + d_yaw,
            x[3] + d_v,
            x[4] + d_steer,
            next_yaw_rate,
            cmd_s
        )

        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = x_next
        model.name = "gem_nn_mpc"
        return model

    def setup_mpc(self):
        if AcadosOcp is None:
            return

        if get_tera is not None:
            try:
                get_tera(tera_version="0.0.34")
            except Exception:
                pass

        model = self.create_acados_model()

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(4)

        if hasattr(ocp.solver_options, "N_horizon"):
            ocp.solver_options.N_horizon = HORIZON
        else:
            ocp.dims.N = HORIZON

        nx, nu = 7, 2
        self.nx, self.nu = nx, nu

        # Weights
        Q_lat = 1.0
        Q_yaw = 1.0
        Q_vel = 5.0

        R_accel = 1.0
        R_steer = 20.0
        R_rate = 2000.0

        Q_steer_state = 0.0

        x_sym = model.x
        u_sym = model.u
        p_sym = model.p

        e_pos = (x_sym[0] - p_sym[0])**2 + (x_sym[1] - p_sym[1])**2
        e_v = x_sym[3] - p_sym[3]
        diff_steer = u_sym[1] - x_sym[6]
        diff_v = u_sym[0] - x_sym[3]
        yaw_diff = x_sym[2] - p_sym[2]

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        if USE_YAW_SINCOS_COST:
            e_sin = ca.sin(yaw_diff)
            e_cos = 1 - ca.cos(yaw_diff)

            y_expr = ca.vertcat(
                e_pos, e_sin, e_cos, e_v,
                diff_v, u_sym[1], x_sym[4], diff_steer
            )
            ocp.model.cost_y_expr = y_expr

            self.ny = 8
            ocp.cost.yref = np.zeros(self.ny)
            ocp.cost.W = np.diag([Q_lat, Q_yaw, Q_yaw, Q_vel, R_accel, R_steer, Q_steer_state, R_rate])

            y_expr_e = ca.vertcat(e_pos, e_sin, e_cos, e_v, x_sym[4])
            ocp.model.cost_y_expr_e = y_expr_e

            self.ny_e = 5
            ocp.cost.yref_e = np.zeros(self.ny_e)
            ocp.cost.W_e = np.diag([Q_lat, Q_yaw, Q_yaw, Q_vel, Q_steer_state])
        else:
            e_yaw = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))

            y_expr = ca.vertcat(e_pos, e_yaw, e_v, diff_v, u_sym[1], x_sym[4], diff_steer)
            ocp.model.cost_y_expr = y_expr

            self.ny = 7
            ocp.cost.yref = np.zeros(self.ny)
            ocp.cost.W = np.diag([Q_lat, Q_yaw, Q_vel, R_accel, R_steer, Q_steer_state, R_rate])

            y_expr_e = ca.vertcat(e_pos, e_yaw, e_v, x_sym[4])
            ocp.model.cost_y_expr_e = y_expr_e

            self.ny_e = 4
            ocp.cost.yref_e = np.zeros(self.ny_e)
            ocp.cost.W_e = np.diag([Q_lat, Q_yaw, Q_vel, Q_steer_state])

        # State bounds: v, steer
        ocp.constraints.idxbx = np.array([3, 4])
        ocp.constraints.lbx = np.array([0.0, -0.65])
        ocp.constraints.ubx = np.array([MAX_REF_V, 0.65])

        # Soft constraints for those bounds
        ocp.constraints.idxsbx = np.array([0, 1])
        ns = 2
        ocp.cost.zl = 1000 * np.ones((ns,))
        ocp.cost.zu = 1000 * np.ones((ns,))
        ocp.cost.Zl = 100 * np.ones((ns,))
        ocp.cost.Zu = 100 * np.ones((ns,))

        # Input bounds
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([0.0, -0.6])
        ocp.constraints.ubu = np.array([MAX_REF_V, 0.6])

        ocp.constraints.x0 = np.zeros(nx)

        # Hard constraint: steering rate using last_cmd_steer state (x[6])
        ocp.constraints.C = np.zeros((1, nx))
        ocp.constraints.C[0, 6] = -1.0
        ocp.constraints.D = np.zeros((1, nu))
        ocp.constraints.D[0, 1] = 1.0

        d_max = MAX_STEER_RATE * DT
        ocp.constraints.lg = np.array([-d_max])
        ocp.constraints.ug = np.array([d_max])

        # Solver options
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.tf = HORIZON * DT
        ocp.solver_options.print_level = 0

        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    # ============================
    # Warm-start rollout (Option A midpoint yaw)
    # ============================
    def build_initial_guess(self, ref_traj, use_prev=True):
        curr_phys = self.current_state
        curr_full = np.append(curr_phys, self.last_cmd_steer)

        if use_prev and self.prev_u_opt is not None:
            u_guess = np.hstack((self.prev_u_opt[:, 1:], self.prev_u_opt[:, -1:]))
        else:
            u_guess = np.zeros((2, HORIZON))
            u_guess[0, :] = curr_phys[3]
            tgt = ref_traj[:, 0]
            heading = np.arctan2(tgt[1] - curr_phys[1], tgt[0] - curr_phys[0])
            err = self.wrap_angle(heading - curr_phys[2])
            u_guess[1, :] = np.clip(err, -0.5, 0.5)

        x_guess = np.zeros((7, HORIZON + 1))
        x_guess[:, 0] = curr_full
        temp_state = curr_full.copy()

        for k in range(HORIZON):
            v, st, yr = temp_state[3], temp_state[4], temp_state[5]
            cv, cs = u_guess[0, k], u_guess[1, k]

            inp = np.array([v, st, yr, cv, cs, DT], dtype=float)
            inp_n = (inp - self.sx_mean) / self.sx_scale

            h1 = np.tanh(self.weights['net.0.weight'].dot(inp_n) + self.weights['net.0.bias'])
            h2 = np.tanh(self.weights['net.2.weight'].dot(h1) + self.weights['net.2.bias'])
            h3 = np.tanh(self.weights['net.4.weight'].dot(h2) + self.weights['net.4.bias'])
            out_n = self.weights['net.6.weight'].dot(h3) + self.weights['net.6.bias']
            out_r = out_n * self.sy_scale + self.sy_mean

            d_yaw = float(out_r[2])
            yaw_mid = temp_state[2] + 0.5 * d_yaw

            dx = np.cos(yaw_mid) * out_r[0] - np.sin(yaw_mid) * out_r[1]
            dy = np.sin(yaw_mid) * out_r[0] + np.cos(yaw_mid) * out_r[1]

            dv = float(out_r[3])
            dsteer = float(out_r[4])

            next_st = np.zeros(7)
            next_st[0] = temp_state[0] + dx
            next_st[1] = temp_state[1] + dy
            next_st[2] = temp_state[2] + d_yaw  # keep continuous
            next_st[3] = temp_state[3] + dv
            next_st[4] = temp_state[4] + dsteer
            next_st[5] = d_yaw / DT
            next_st[6] = cs

            x_guess[:, k + 1] = next_st
            temp_state = next_st

        return x_guess, u_guess

    def compensate_latency(self, x_curr):
        dt = 0.13
        x, y, yaw, v, steer, yaw_rate = x_curr
        nx = x + v * np.cos(yaw) * dt
        ny = y + v * np.sin(yaw) * dt
        nyaw = yaw + yaw_rate * dt  # continuous
        return np.array([nx, ny, nyaw, v, steer, yaw_rate], dtype=float)

    def solve_with_guess(self, ref_traj, use_prev=True):
        try:
            x_guess, u_guess = self.build_initial_guess(ref_traj, use_prev)

            x0_lat = self.compensate_latency(self.current_state)
            x0 = np.append(x0_lat, self.last_cmd_steer)

            self.solver.set(0, 'lbx', x0)
            self.solver.set(0, 'ubx', x0)

            for k in range(HORIZON):
                self.solver.set(k, 'yref', np.zeros(self.ny))
                self.solver.set(k, 'p', ref_traj[:4, k])
                self.solver.set(k, 'x', x_guess[:, k])
                self.solver.set(k, 'u', u_guess[:, k])

            self.solver.set(HORIZON, 'p', ref_traj[:4, -1])
            self.solver.set(HORIZON, 'x', x_guess[:, -1])

            status = self.solver.solve()
            if status != 0:
                return None

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

    # ============================
    # Callbacks
    # ============================
    def state_callback(self, msg):
        try:
            idx = msg.name.index("gem")
            p = msg.pose[idx].position
            q = msg.pose[idx].orientation
            v_lin = msg.twist[idx].linear
            w_ang = msg.twist[idx].angular

            (_, _, yaw_meas) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            yaw_meas = self.wrap_angle(float(yaw_meas))
            self._yaw_meas_wrapped = yaw_meas

            if self._yaw_cont is None:
                self._yaw_cont = yaw_meas
                self._yaw_meas_prev = yaw_meas
            else:
                raw_jump = yaw_meas - self._yaw_meas_prev
                dyaw = self.wrap_angle(yaw_meas - self._yaw_meas_prev)

                if RESET_WARMSTART_ON_YAW_WRAP_JUMP and abs(raw_jump) > YAW_WRAP_JUMP_THRESH:
                    self.prev_u_opt = None
                    self.prev_x_opt = None
                    rospy.logwarn(f"Yaw wrap jump detected: raw_jump={raw_jump:.3f} rad. Clearing warm start.")

                self._yaw_cont += dyaw
                self._yaw_meas_prev = yaw_meas

            speed = float(np.sqrt(v_lin.x**2 + v_lin.y**2))

            self.current_state = np.array(
                [float(p.x), float(p.y), float(self._yaw_cont), speed, float(self._steer_data), float(w_ang.z)],
                dtype=float
            )
            self._yaw_rate_data = float(w_ang.z)
        except ValueError:
            pass

    def joint_state_callback(self, msg):
        try:
            if self._left_idx is None:
                self._left_idx = msg.name.index("left_steering_hinge_joint")
                self._right_idx = msg.name.index("right_steering_hinge_joint")
            l = float(msg.position[self._left_idx])
            r = float(msg.position[self._right_idx])
            self._left_steer = l
            self._right_steer = r
            self._steer_data = 0.5 * (l + r)
        except ValueError:
            pass

    def imu_callback(self, msg):
        self.imu_data['ax'] = msg.linear_acceleration.x
        self.imu_data['ay'] = msg.linear_acceleration.y
        self.imu_data['az'] = msg.linear_acceleration.z
        self.imu_data['wz'] = msg.angular_velocity.z

    # ============================
    # Helpers
    # ============================
    @staticmethod
    def wrap_angle(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def index_ahead_by_distance(self, start_idx, dist):
        if len(self.path_s) == 0 or self.total_path_length <= 1e-9:
            return 0
        t_s = (self.path_s[start_idx] + dist) % self.total_path_length
        idx = np.searchsorted(self.path_s, t_s)
        return min(int(idx), len(self.path_s) - 1)

    def calc_closest_index(self, path, state):
        dx = path[:, COL_X] - state[0]
        dy = path[:, COL_Y] - state[1]
        d = np.hypot(dx, dy)
        min_idx = int(np.argmin(d))
        min_dist = float(d[min_idx])
        return min_idx, min_dist

    # ============================
    # Reference trajectory (yaw unwrapped along horizon, anchored to continuous yaw)
    # ============================
    def get_reference_trajectory(self):
        if self.current_state is None:
            return None

        cx, cy, cyaw = float(self.current_state[0]), float(self.current_state[1]), float(self.current_state[2])
        min_idx, _ = self.calc_closest_index(self.full_path, self.current_state)

        dx_err = cx - float(self.full_path[min_idx, COL_X])
        dy_err = cy - float(self.full_path[min_idx, COL_Y])
        path_yaw_wrapped = float(self.full_path[min_idx, COL_YAW])
        cte_signed = (-np.sin(path_yaw_wrapped) * dx_err + np.cos(path_yaw_wrapped) * dy_err)
        self.cte_history.append(float(cte_signed))

        traj = np.zeros((HORIZON, 7), dtype=float)
        prev_yaw_ref = None

        if self.spline_tck is None or self.spline_u_pts is None:
            idx = min_idx
            for i in range(HORIZON):
                goal_pt = self.full_path[idx]
                x_ref = float(goal_pt[COL_X])
                y_ref = float(goal_pt[COL_Y])
                yaw_raw = float(goal_pt[COL_YAW])
                v_ref = float(goal_pt[COL_V])

                if prev_yaw_ref is None:
                    yaw_ref = cyaw + self.wrap_angle(yaw_raw - cyaw)
                else:
                    yaw_ref = prev_yaw_ref + self.wrap_angle(yaw_raw - prev_yaw_ref)
                prev_yaw_ref = yaw_ref

                traj[i, 0] = x_ref
                traj[i, 1] = y_ref
                traj[i, 2] = yaw_ref
                traj[i, 3] = v_ref

                dist_step = max(0.2, max(float(self.current_state[3]), 0.1) * DT)
                idx = self.index_ahead_by_distance(idx, dist_step)
        else:
            current_u = float(self.spline_u_pts[min_idx])
            tck = self.spline_tck

            for i in range(HORIZON):
                x_ref, y_ref = splev(current_u, tck)
                dx, dy = splev(current_u, tck, der=1)
                norm_deriv = float(np.hypot(dx, dy))
                yaw_raw = float(np.arctan2(dy, dx))

                if prev_yaw_ref is None:
                    yaw_ref = cyaw + self.wrap_angle(yaw_raw - cyaw)
                else:
                    yaw_ref = prev_yaw_ref + self.wrap_angle(yaw_raw - prev_yaw_ref)
                prev_yaw_ref = yaw_ref

                ddx, ddy = splev(current_u, tck, der=2)
                curvature = float((dx * ddy - dy * ddx) / (norm_deriv**3 + 1e-6))

                max_lat_accel = 0.4
                v_limit = np.sqrt(max_lat_accel / (abs(curvature) + 1e-3))
                v_ref = min(MAX_REF_V, float(v_limit))
                v_ref = max(v_ref, MIN_REF_V)

                traj[i, 0] = float(x_ref)
                traj[i, 1] = float(y_ref)
                traj[i, 2] = float(yaw_ref)
                traj[i, 3] = float(v_ref)

                ds = v_ref * DT
                du = ds / (norm_deriv + 1e-6)
                current_u += du

                if current_u > float(self.spline_u_pts[-1]):
                    current_u = float(self.spline_u_pts[-1])

        return traj.T

    # ============================
    # Main loop
    # ============================
    def run(self):
        if not self.initialized:
            return

        while not rospy.is_shutdown():
            if self.current_state is None:
                self.rate.sleep()
                continue

            ref_traj = self.get_reference_trajectory()
            if ref_traj is None:
                self.rate.sleep()
                continue

            use_warm = (self.prev_u_opt is not None)
            res = self.solve_with_guess(ref_traj, use_warm)
            if res is None and use_warm:
                res = self.solve_with_guess(ref_traj, False)

            msg = AckermannDrive()
            steer_cmd_opt = 0.0
            ref_yaw0 = float(ref_traj[2, 0])

            if res is not None:
                x_opt, u_opt = res
                self.prev_u_opt = u_opt
                self.prev_x_opt = x_opt

                cmd_v = float(u_opt[0, 0])
                cmd_s = float(u_opt[1, 0])
                steer_cmd_opt = cmd_s
                self.last_cmd_steer = cmd_s

                # Output filter
                ALPHA = 0.9
                filtered_cmd = ALPHA * cmd_s + (1 - ALPHA) * self.last_pub_steer
                self.last_pub_steer = filtered_cmd

                # Apply gain/offset (published)
                final_pub_steer = (filtered_cmd / STEER_GAIN_COMP) + STEER_OFFSET

                msg.speed = cmd_v
                msg.steering_angle = final_pub_steer

                if self.vis_pub.get_num_connections() > 0:
                    self.publish_markers(ref_traj, x_opt)
            else:
                msg.speed = 0.0
                msg.steering_angle = 0.0
                rospy.logwarn("Acados solve failed -> stopping")

            self.pub.publish(msg)

            # LOGGING (include x,y,yaw for RMSE evaluation later)
            try:
                t = float(rospy.get_time())
                cte = float(self.cte_history[-1]) if self.cte_history else 0.0

                sx, sy, syaw, sv = float(self.current_state[0]), float(self.current_state[1]), float(self.current_state[2]), float(self.current_state[3])

                row = [
                    t, cte,
                    float(msg.speed),
                    float(msg.steering_angle),
                    float(steer_cmd_opt),
                    sv,
                    float(self._steer_data),
                    float(self._yaw_rate_data),

                    sx, sy, syaw,
                    float(self._yaw_meas_wrapped), ref_yaw0,

                    float(self._left_steer), float(self._right_steer),
                    float(self.imu_data['ax']), float(self.imu_data['ay']), float(self.imu_data['az']), float(self.imu_data['wz']),
                ]
                self.log_data.append(row)

                if res is not None:
                    self.traj_log.append({'time': t, 'ref': ref_traj, 'pred': res[0]})
            except Exception:
                pass

            self.rate.sleep()

    # ============================
    # Visualization
    # ============================
    def publish_markers(self, ref_traj, x_opt):
        if self.current_state is None:
            return

        cx, cy, cyaw = self.current_state[:3]
        cos_y, sin_y = np.cos(cyaw), np.sin(cyaw)

        def g2l(gx, gy):
            dx, dy = gx - cx, gy - cy
            return (cos_y * dx + sin_y * dy), (-sin_y * dx + cos_y * dy)

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
        m1.scale.x = 0.15
        m1.scale.y = 0.15
        m1.color.a = 1.0
        m1.color.g = 1.0

        for i in range(ref_traj.shape[1]):
            lx, ly = g2l(ref_traj[0, i], ref_traj[1, i])
            m1.points.append(Point(x=float(lx), y=float(ly), z=-0.5))
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
            m2.scale.x = 0.1
            m2.scale.y = 0.1
            m2.scale.z = 0.1
            m2.color.a = 1.0
            m2.color.b = 1.0

            for i in range(x_opt.shape[1]):
                lx, ly = g2l(x_opt[0, i], x_opt[1, i])
                if np.isnan(lx) or np.isnan(ly):
                    continue
                m2.points.append(Point(x=float(lx), y=float(ly), z=-0.5))
            ma.markers.append(m2)

        self.vis_pub.publish(ma)

    # ============================
    # Teleport
    # ============================
    def teleport_to_start(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            sx = float(self.full_path[0, COL_X])
            sy = float(self.full_path[0, COL_Y])
            syaw = float(self.full_path[0, COL_YAW])

            s = ModelState()
            s.model_name = 'gem'
            s.pose.position.x = sx
            s.pose.position.y = sy
            s.pose.position.z = 2.0

            q = quaternion_from_euler(0, 0, syaw)
            s.pose.orientation.x = q[0]
            s.pose.orientation.y = q[1]
            s.pose.orientation.z = q[2]
            s.pose.orientation.w = q[3]

            rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(s)
            self.pub.publish(AckermannDrive())

            # Initialize yaw unwrap baseline
            syaw_wrapped = self.wrap_angle(syaw)
            self._yaw_meas_prev = syaw_wrapped
            self._yaw_cont = float(syaw)  # continuous baseline
            self._yaw_meas_wrapped = syaw_wrapped
        except Exception:
            pass


if __name__ == '__main__':
    c = NeuralMPC()
    try:
        c.run()
    except rospy.ROSInterruptException:
        pass
