#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import pandas as pd
import torch
import casadi as ca
import pickle
from scipy.interpolate import splprep, splev
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, get_tera

from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Plotting (headless safe)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================
# CONFIG
# ============================
MODEL_PATH = "gem_dynamics.pth"
SCALER_PATH = "gem_scaler.pkl"
SCALER_ARRAY_PATH = "gem_scaler_arrays.npz"
PATH_CSV = "wps.csv"

if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core

HORIZON = 20
DT = 0.05

MIN_REF_V = 0.5
MAX_REF_V = 5.5

MAX_ACCEL = 2.0          # m/s^2 (hard cmd_v rate constraint)
MAX_STEER_RATE = 0.5     # rad/s (hard cmd_steer rate constraint)

ALPHA_STEER = 0.90
ALPHA_V = 0.80

VREF_RAMP_ACCEL = 1.0    # m/s^2 max change of v_ref
VADV_EXTRA = 0.3         # max lookahead speed boost over measured v

STEER_GAIN_COMP = 1.0
STEER_OFFSET = 0.0

# RMSE / plots
RMSE_PLOT_PATH = "mpc_rmse.png"
CTE_PLOT_PATH = "mpc_cte_signed.png"

# online sysid evaluation tolerance
EVAL_DT_TOL = 0.02       # accept dt_meas in [DT - tol, DT + tol]

COL_X = 0
COL_Y = 1
COL_YAW = 2
COL_V = 4


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class NeuralMPC:
    def __init__(self):
        rospy.init_node("neural_mpc_acados")

        # Measured physical state: [x, y, yaw_unwrapped, v, steer, yaw_rate]
        self.current_state = None

        # yaw unwrap memory
        self._yaw_wrapped_prev = None
        self._yaw_unwrapped = None

        # command memory (augmented states)
        self.last_cmd_steer = 0.0
        self.last_cmd_v = 0.0

        # filtered published
        self.last_pub_steer = 0.0
        self.last_pub_v = 0.0

        # warm start
        self.prev_u_opt = None
        self.prev_x_opt = None

        self.rate = rospy.Rate(1.0 / DT)
        self.initialized = False

        # logs
        self.log_data = []
        self.traj_log = []

        # online sysid eval
        self._eval_prev = None  # {t, state6, cmd_v, cmd_s, prev_cmd_v, prev_cmd_s}
        self.eval_err_log = []  # rows: [t, e_dx, e_dy, e_dyaw, e_dv, e_dsteer]
        self.cte_log = []       # rows: [t, cte_signed]

        # sensors
        self._steer_data = 0.0
        self._left_steer = 0.0
        self._right_steer = 0.0
        self._yaw_rate_data = 0.0
        self._left_idx = None
        self._right_idx = None
        self.imu_data = {"ax": 0, "ay": 0, "az": 0, "wz": 0}
        self._vref_last = None

        if not self.load_scalers():
            rospy.logerr("Failed to load scalers.")
            return

        # Expect Option A input dim = 8, output dim = 5
        if len(self.sx_mean) != 8 or len(self.sx_scale) != 8:
            rospy.logerr(f"Scaler X dim mismatch. Expected 8, got {len(self.sx_mean)}")
            return
        if len(self.sy_mean) != 5 or len(self.sy_scale) != 5:
            rospy.logerr(f"Scaler Y dim mismatch. Expected 5, got {len(self.sy_mean)}")
            return

        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        self.weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
        print("Weights loaded.")

        # LOAD PATH
        try:
            df = pd.read_csv(PATH_CSV, header=None)
            raw_path = df.values

            diff = np.diff(raw_path[:, :2], axis=0)
            dist = np.linalg.norm(diff, axis=1)
            mask = np.concatenate(([True], dist > 1e-4))
            self.full_path = raw_path[mask]

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

        self.pub = rospy.Publisher("/gem/ackermann_cmd", AckermannDrive, queue_size=1)
        self.vis_pub = rospy.Publisher("/gem/mpc_debug", MarkerArray, queue_size=1, latch=True)

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/gem/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/gem/imu", Imu, self.imu_callback)

        self.teleport_to_start()

        print("Neutralizing steering for 3s...")
        z_msg = AckermannDrive()
        z_msg.speed = 0.0
        z_msg.steering_angle = 0.0
        for _ in range(30):
            self.pub.publish(z_msg)
            rospy.sleep(0.1)

        print("Acados MPC Ready.")
        rospy.on_shutdown(self.save_all)
        self.initialized = True

    # ============================
    # SAVE ALL (CSV + plots + traj)
    # ============================
    def save_all(self):
        # CSV
        if self.log_data:
            df = pd.DataFrame(
                self.log_data,
                columns=[
                    "time",
                    "cte_signed",
                    "steer_cmd_pub",
                    "cmd_speed_pub",
                    "steer_act",
                    "yaw_rate",
                    "speed",
                    "ax",
                    "ay",
                    "az",
                    "wz",
                    "last_cmd_v",
                    "last_cmd_steer",
                ],
            )
            df.to_csv("mpc_debug.csv", index=False)
            print("Saved mpc_debug.csv")

        # traj pickle
        if self.traj_log:
            try:
                with open("mpc_trajectories.pkl", "wb") as f:
                    pickle.dump(self.traj_log, f)
                print("Saved mpc_trajectories.pkl")
            except Exception as e:
                print(f"Failed to save trajectories: {e}")

        # RMSE + CTE plots
        self.save_plots()

    def save_plots(self):
        if plt is None:
            print("[WARN] matplotlib not available; skipping RMSE/CTE plots.")
            return

        # --- CTE plot ---
        if self.cte_log:
            t0 = self.cte_log[0][0]
            ts = np.array([r[0] - t0 for r in self.cte_log], dtype=float)
            cte = np.array([r[1] for r in self.cte_log], dtype=float)

            fig = plt.figure(figsize=(12, 5))
            ax = plt.gca()
            ax.plot(ts, cte)
            ax.grid(True)
            ax.set_title("Signed Cross Track Error (CTE)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("cte_signed (m)")
            plt.tight_layout()
            plt.savefig(CTE_PLOT_PATH, dpi=180)
            plt.close(fig)
            print(f"Saved {CTE_PLOT_PATH}")

        # --- RMSE plot ---
        if self.eval_err_log:
            t0 = self.eval_err_log[0][0]
            ts = np.array([r[0] - t0 for r in self.eval_err_log], dtype=float)
            E = np.array([r[1:] for r in self.eval_err_log], dtype=float)  # [N,5]
            E2 = E * E
            cums = np.cumsum(E2, axis=0)
            denom = np.arange(1, len(E) + 1).reshape(-1, 1)
            rmse = np.sqrt(cums / denom)  # running RMSE

            labels = ["dx_local", "dy_local", "d_yaw", "d_v", "d_steer"]
            fig = plt.figure(figsize=(12, 6))
            ax = plt.gca()
            for i in range(5):
                ax.plot(ts, rmse[:, i], label=labels[i])
            ax.grid(True)
            ax.set_title("Online 1-step Model Running RMSE (local deltas)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("RMSE (units of each output)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(RMSE_PLOT_PATH, dpi=180)
            plt.close(fig)
            print(f"Saved {RMSE_PLOT_PATH}")

    # ============================
    # IO
    # ============================
    def load_scalers(self):
        try:
            with open(SCALER_PATH, "rb") as f:
                scalers = pickle.load(f)
                self.sx_mean = np.asarray(scalers["x"].mean_, dtype=float)
                self.sx_scale = np.asarray(scalers["x"].scale_, dtype=float)
                self.sy_mean = np.asarray(scalers["y"].mean_, dtype=float)
                self.sy_scale = np.asarray(scalers["y"].scale_, dtype=float)
            return True
        except Exception:
            try:
                arrs = np.load(SCALER_ARRAY_PATH)
                self.sx_mean = np.asarray(arrs["x_mean"], dtype=float)
                self.sx_scale = np.asarray(arrs["x_scale"], dtype=float)
                self.sy_mean = np.asarray(arrs["y_mean"], dtype=float)
                self.sy_scale = np.asarray(arrs["y_scale"], dtype=float)
                return True
            except Exception:
                return False

    # ============================
    # PATH / SPLINE
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
        k = np.zeros(len(path))
        window = 5
        half = window // 2

        for i in range(half, len(path) - half):
            pts = path[i - half : i + half + 1, :2]
            centroid = np.mean(pts, axis=0)
            pts_c = pts - centroid
            x = pts_c[:, 0]
            y = pts_c[:, 1]

            M = np.column_stack((x, y, np.ones(len(x))))
            D = x**2 + y**2
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
            v_limit = np.sqrt(max_lat_accel / (abs(curvature) + 1e-3))
            path[i, COL_V] = min(MAX_REF_V, v_limit)
            path[i, COL_V] = max(path[i, COL_V], MIN_REF_V)

        print("Velocity Profile Generated based on Curvature.")

    # ============================
    # NN forward (numpy)
    # ============================
    def nn_predict_local_delta(self, v, steer, yaw_rate, cmd_v, cmd_s, dt, prev_cmd_v, prev_cmd_s):
        # Option A input order:
        inp = np.array([v, steer, yaw_rate, cmd_v, cmd_s, dt, prev_cmd_v, prev_cmd_s], dtype=float)
        inp_n = (inp - self.sx_mean) / self.sx_scale

        h1 = np.tanh(self.weights["net.0.weight"].dot(inp_n) + self.weights["net.0.bias"])
        h2 = np.tanh(self.weights["net.2.weight"].dot(h1) + self.weights["net.2.bias"])
        h3 = np.tanh(self.weights["net.4.weight"].dot(h2) + self.weights["net.4.bias"])
        out_n = self.weights["net.6.weight"].dot(h3) + self.weights["net.6.bias"]
        out_r = out_n * self.sy_scale + self.sy_mean
        # out_r = [dx_local, dy_local, d_yaw, d_v, d_steer]
        return out_r

    # ============================
    # ACADOS MODEL (Option A)
    # ============================
    def create_acados_model(self):
        model = AcadosModel()

        # x = [X, Y, Yaw(unwrapped), v, steer, yaw_rate, prev_cmd_steer, prev_cmd_v]
        x = ca.SX.sym("x", 8)
        u = ca.SX.sym("u", 2)
        p = ca.SX.sym("p", 4)

        yaw = x[2]
        v_act = x[3]
        steer_act = x[4]
        yaw_rate = x[5]
        prev_cmd_steer = x[6]
        prev_cmd_v = x[7]

        cmd_v = u[0]
        cmd_s = u[1]

        # NN input: [v, steer, yaw_rate, cmd_v, cmd_s, dt, prev_cmd_v, prev_cmd_steer]
        nn_input = ca.vertcat(v_act, steer_act, yaw_rate, cmd_v, cmd_s, DT, prev_cmd_v, prev_cmd_steer)

        sx_mean = ca.DM(self.sx_mean)
        sx_scale = ca.DM(self.sx_scale)
        sy_mean = ca.DM(self.sy_mean)
        sy_scale = ca.DM(self.sy_scale)

        inp_norm = (nn_input - sx_mean) / sx_scale

        act = ca.tanh
        w1 = ca.DM(self.weights["net.0.weight"]); b1 = ca.DM(self.weights["net.0.bias"])
        h1 = act(ca.mtimes(w1, inp_norm) + b1)
        w2 = ca.DM(self.weights["net.2.weight"]); b2 = ca.DM(self.weights["net.2.bias"])
        h2 = act(ca.mtimes(w2, h1) + b2)
        w3 = ca.DM(self.weights["net.4.weight"]); b3 = ca.DM(self.weights["net.4.bias"])
        h3 = act(ca.mtimes(w3, h2) + b3)
        w4 = ca.DM(self.weights["net.6.weight"]); b4 = ca.DM(self.weights["net.6.bias"])
        out_norm = ca.mtimes(w4, h3) + b4

        out_real = out_norm * sy_scale + sy_mean
        dx_l = out_real[0]
        dy_l = out_real[1]
        d_yaw = out_real[2]
        d_v = out_real[3]
        d_steer = out_real[4]

        # Mid-yaw rotation (matches training)
        yaw_mid = yaw + 0.5 * d_yaw
        dx_g = ca.cos(yaw_mid) * dx_l - ca.sin(yaw_mid) * dy_l
        dy_g = ca.sin(yaw_mid) * dx_l + ca.cos(yaw_mid) * dy_l

        next_yaw = yaw + d_yaw
        next_v = v_act + d_v
        next_steer = steer_act + d_steer
        next_yaw_rate = d_yaw / DT

        x_next = ca.vertcat(
            x[0] + dx_g,
            x[1] + dy_g,
            next_yaw,
            next_v,
            next_steer,
            next_yaw_rate,
            cmd_s,  # prev_cmd_steer next
            cmd_v,  # prev_cmd_v next
        )

        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = x_next
        model.name = "gem_nn_mpc"
        return model

    def setup_mpc(self):
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

        nx = 8
        nu = 2

        # Cost weights
        Q_lat = 1.0
        Q_yaw = 1.0
        Q_vel = 5.0

        R_track_v = 1.0
        R_steer = 20.0
        R_steer_rate = 2000.0
        R_cmdv_rate = 50.0

        x_sym = model.x
        u_sym = model.u
        p_sym = model.p

        e_pos = (x_sym[0] - p_sym[0])**2 + (x_sym[1] - p_sym[1])**2
        yaw_diff = x_sym[2] - p_sym[2]
        e_yaw = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))
        e_v = x_sym[3] - p_sym[3]

        diff_v_track = u_sym[0] - x_sym[3]
        diff_steer_rate = u_sym[1] - x_sym[6]
        diff_cmdv_rate = u_sym[0] - x_sym[7]

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        y_expr = ca.vertcat(
            e_pos,
            e_yaw,
            e_v,
            diff_v_track,
            u_sym[1],
            x_sym[4],
            diff_steer_rate,
            diff_cmdv_rate,
        )
        ocp.model.cost_y_expr = y_expr
        ocp.cost.W = np.diag([Q_lat, Q_yaw, Q_vel, R_track_v, R_steer, 0.0, R_steer_rate, R_cmdv_rate])
        ocp.cost.yref = np.zeros(8)

        y_expr_e = ca.vertcat(e_pos, e_yaw, e_v, x_sym[4])
        ocp.model.cost_y_expr_e = y_expr_e
        ocp.cost.W_e = np.diag([Q_lat, Q_yaw, Q_vel, 0.0])
        ocp.cost.yref_e = np.zeros(4)

        # bounds
        ocp.constraints.idxbx = np.array([3, 4])
        ocp.constraints.lbx = np.array([0.0, -0.65])
        ocp.constraints.ubx = np.array([MAX_REF_V, 0.65])

        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([0.0, -0.6])
        ocp.constraints.ubu = np.array([MAX_REF_V, 0.6])

        ocp.constraints.x0 = np.zeros(nx)

        # general constraints (steer rate + accel rate)
        ocp.constraints.C = np.zeros((2, nx))
        ocp.constraints.D = np.zeros((2, nu))

        # steer: u_s - prev_cmd_steer
        ocp.constraints.C[0, 6] = -1.0
        ocp.constraints.D[0, 1] = 1.0
        d_max = MAX_STEER_RATE * DT

        # accel: u_v - prev_cmd_v
        ocp.constraints.C[1, 7] = -1.0
        ocp.constraints.D[1, 0] = 1.0
        a_max = MAX_ACCEL * DT

        ocp.constraints.lg = np.array([-d_max, -a_max])
        ocp.constraints.ug = np.array([ d_max,  a_max])

        # solver
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.tf = HORIZON * DT
        ocp.solver_options.print_level = 0

        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    # ============================
    # INITIAL GUESS ROLLOUT
    # ============================
    def build_initial_guess(self, ref_traj, use_prev=True):
        curr_phys = self.current_state  # 6D
        curr_full = np.concatenate([curr_phys, [self.last_cmd_steer, self.last_cmd_v]])  # 8D

        if use_prev and self.prev_u_opt is not None:
            u_guess = np.hstack((self.prev_u_opt[:, 1:], self.prev_u_opt[:, -1:]))
        else:
            u_guess = np.zeros((2, HORIZON))
            u_guess[0, :] = max(curr_phys[3], 0.0)

            tgt = ref_traj[:, 0]
            heading = np.arctan2(tgt[1] - curr_phys[1], tgt[0] - curr_phys[0])
            err = wrap_angle(heading - curr_phys[2])
            u_guess[1, :] = np.clip(err, -0.5, 0.5)

        x_guess = np.zeros((8, HORIZON + 1))
        x_guess[:, 0] = curr_full
        temp_state = curr_full.copy()

        for k in range(HORIZON):
            v = temp_state[3]
            st = temp_state[4]
            yr = temp_state[5]
            prev_cmd_steer = temp_state[6]
            prev_cmd_v = temp_state[7]

            cv = u_guess[0, k]
            cs = u_guess[1, k]

            out_r = self.nn_predict_local_delta(v, st, yr, cv, cs, DT, prev_cmd_v, prev_cmd_steer)
            dx_l, dy_l, d_yaw, d_v, d_steer = out_r

            yaw = temp_state[2]
            yaw_mid = yaw + 0.5 * d_yaw
            dx = np.cos(yaw_mid) * dx_l - np.sin(yaw_mid) * dy_l
            dy = np.sin(yaw_mid) * dx_l + np.cos(yaw_mid) * dy_l

            next_st = np.zeros(8)
            next_st[0] = temp_state[0] + dx
            next_st[1] = temp_state[1] + dy
            next_st[2] = temp_state[2] + d_yaw
            next_st[3] = temp_state[3] + d_v
            next_st[4] = temp_state[4] + d_steer
            next_st[5] = d_yaw / DT
            next_st[6] = cs
            next_st[7] = cv

            x_guess[:, k + 1] = next_st
            temp_state = next_st

        return x_guess, u_guess

    def compensate_latency(self, x_curr):
        dt = 0.13
        x, y, yaw, v, steer, yaw_rate = x_curr
        nx = x + v * np.cos(yaw) * dt
        ny = y + v * np.sin(yaw) * dt
        nyaw = yaw + yaw_rate * dt
        return np.array([nx, ny, nyaw, v, steer, yaw_rate], dtype=float)

    def solve_with_guess(self, ref_traj, use_prev=True):
        try:
            x_guess, u_guess = self.build_initial_guess(ref_traj, use_prev)

            x0_lat = self.compensate_latency(self.current_state)
            x0 = np.concatenate([x0_lat, [self.last_cmd_steer, self.last_cmd_v]])  # 8D

            self.solver.set(0, "lbx", x0)
            self.solver.set(0, "ubx", x0)

            for k in range(HORIZON):
                self.solver.set(k, "yref", np.zeros(8))
                self.solver.set(k, "p", ref_traj[:4, k])
                self.solver.set(k, "x", x_guess[:, k])
                self.solver.set(k, "u", u_guess[:, k])

            self.solver.set(HORIZON, "p", ref_traj[:4, -1])
            self.solver.set(HORIZON, "x", x_guess[:, -1])

            status = self.solver.solve()
            if status != 0:
                return None

            u_opt = np.zeros((2, HORIZON))
            x_opt = np.zeros((8, HORIZON + 1))
            for k in range(HORIZON):
                u_opt[:, k] = self.solver.get(k, "u")
                x_opt[:, k] = self.solver.get(k, "x")
            x_opt[:, HORIZON] = self.solver.get(HORIZON, "x")

            return x_opt, u_opt
        except Exception:
            if use_prev:
                self.prev_u_opt = None
                self.prev_x_opt = None
            return None

    # ============================
    # ROS CALLBACKS
    # ============================
    def state_callback(self, msg):
        try:
            idx = msg.name.index("gem")
            p = msg.pose[idx].position
            q = msg.pose[idx].orientation
            v_lin = msg.twist[idx].linear
            w_ang = msg.twist[idx].angular

            (_, _, yaw_wrapped) = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # unwrap yaw
            if self._yaw_wrapped_prev is None:
                self._yaw_wrapped_prev = yaw_wrapped
                self._yaw_unwrapped = yaw_wrapped
            else:
                dy = wrap_angle(yaw_wrapped - self._yaw_wrapped_prev)
                self._yaw_unwrapped += dy
                self._yaw_wrapped_prev = yaw_wrapped

            speed = float(np.sqrt(v_lin.x**2 + v_lin.y**2))

            self.current_state = np.array(
                [p.x, p.y, float(self._yaw_unwrapped), speed, self._steer_data, float(w_ang.z)],
                dtype=float,
            )
            self._yaw_rate_data = float(w_ang.z)
        except ValueError:
            pass

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
        except ValueError:
            pass

    def imu_callback(self, msg):
        self.imu_data["ax"] = msg.linear_acceleration.x
        self.imu_data["ay"] = msg.linear_acceleration.y
        self.imu_data["az"] = msg.linear_acceleration.z
        self.imu_data["wz"] = msg.angular_velocity.z

    # ============================
    # CTE
    # ============================
    def calc_closest_index(self, path, state):
        dx = path[:, COL_X] - state[0]
        dy = path[:, COL_Y] - state[1]
        d = np.hypot(dx, dy)
        min_idx = int(np.argmin(d))
        return min_idx, float(d[min_idx])

    def signed_cte(self, state):
        idx, _ = self.calc_closest_index(self.full_path, state)
        px, py = self.full_path[idx, COL_X], self.full_path[idx, COL_Y]
        path_yaw = float(self.full_path[idx, COL_YAW])

        dx = state[0] - px
        dy = state[1] - py

        # normal = [-sin(yaw), cos(yaw)]
        cte = (-np.sin(path_yaw) * dx + np.cos(path_yaw) * dy)
        return float(cte), idx

    # ============================
    # ONLINE SYSID EVAL (1-step)
    # ============================
    def try_eval_one_step(self, t_now, state_now):
        if self._eval_prev is None:
            return

        prev = self._eval_prev
        dt_meas = float(t_now - prev["t"])
        if not (DT - EVAL_DT_TOL <= dt_meas <= DT + EVAL_DT_TOL):
            return

        s0 = prev["state6"]
        s1 = state_now

        # Ground truth deltas (local frame, midpoint yaw rotation)
        dx_g = float(s1[0] - s0[0])
        dy_g = float(s1[1] - s0[1])

        d_yaw = wrap_angle(float(s1[2] - s0[2]))
        yaw_mid = float(s0[2] + 0.5 * d_yaw)

        dx_l = np.cos(yaw_mid) * dx_g + np.sin(yaw_mid) * dy_g
        dy_l = -np.sin(yaw_mid) * dx_g + np.cos(yaw_mid) * dy_g

        d_v = float(s1[3] - s0[3])
        d_steer = float(s1[4] - s0[4])

        y_true = np.array([dx_l, dy_l, d_yaw, d_v, d_steer], dtype=float)

        # Prediction uses the same input as MPC model (Option A)
        v0, steer0, yr0 = float(s0[3]), float(s0[4]), float(s0[5])
        cmd_v0 = float(prev["cmd_v"])
        cmd_s0 = float(prev["cmd_s"])
        prev_cmd_v0 = float(prev["prev_cmd_v"])
        prev_cmd_s0 = float(prev["prev_cmd_s"])

        y_pred = self.nn_predict_local_delta(v0, steer0, yr0, cmd_v0, cmd_s0, DT, prev_cmd_v0, prev_cmd_s0)

        e = (y_pred - y_true).astype(float)
        self.eval_err_log.append([t_now, e[0], e[1], e[2], e[3], e[4]])

    # ============================
    # REFERENCE TRAJ
    # ============================
    def get_reference_trajectory(self):
        if self.current_state is None:
            return None

        min_idx, _ = self.calc_closest_index(self.full_path, self.current_state)

        traj = np.zeros((HORIZON, 4), dtype=float)

        if self.spline_tck is None or self.spline_u_pts is None:
            idx = min_idx
            for i in range(HORIZON):
                goal = self.full_path[idx]
                traj[i, 0] = goal[COL_X]
                traj[i, 1] = goal[COL_Y]
                traj[i, 2] = goal[COL_YAW]
                traj[i, 3] = goal[COL_V]
                idx = min(idx + 1, len(self.full_path) - 1)
            return traj.T

        tck = self.spline_tck
        current_u = float(self.spline_u_pts[min_idx])

        for i in range(HORIZON):
            x_ref, y_ref = splev(current_u, tck)
            dx, dy = splev(current_u, tck, der=1)
            norm_deriv = float(np.hypot(dx, dy))
            yaw_ref = float(np.arctan2(dy, dx))

            ddx, ddy = splev(current_u, tck, der=2)
            curvature = float((dx * ddy - dy * ddx) / (norm_deriv**3 + 1e-6))

            max_lat_accel = 0.4
            v_limit = float(np.sqrt(max_lat_accel / (abs(curvature) + 1e-3)))
            v_ref = float(np.clip(v_limit, MIN_REF_V, MAX_REF_V))

            # ramp v_ref
            if self._vref_last is None:
                self._vref_last = v_ref
            dv_allow = VREF_RAMP_ACCEL * DT
            v_ref = float(np.clip(v_ref, self._vref_last - dv_allow, self._vref_last + dv_allow))
            self._vref_last = v_ref

            traj[i, 0] = float(x_ref)
            traj[i, 1] = float(y_ref)
            traj[i, 2] = yaw_ref
            traj[i, 3] = v_ref

            v_adv = min(v_ref, float(self.current_state[3]) + VADV_EXTRA)
            ds = v_adv * DT
            du = ds / (norm_deriv + 1e-6)
            current_u += du
            if current_u > float(self.spline_u_pts[-1]):
                current_u = float(self.spline_u_pts[-1])

        return traj.T

    # ============================
    # MARKERS
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
            m1.points.append(Point(x=lx, y=ly, z=-0.5))
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
                m2.points.append(Point(x=lx, y=ly, z=-0.5))
            ma.markers.append(m2)

        self.vis_pub.publish(ma)

    # ============================
    # TELEPORT
    # ============================
    def teleport_to_start(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            sx = float(self.full_path[0, COL_X])
            sy = float(self.full_path[0, COL_Y])
            syaw = float(self.full_path[0, COL_YAW])

            s = ModelState()
            s.model_name = "gem"
            s.pose.position.x = sx
            s.pose.position.y = sy
            s.pose.position.z = 2.0
            q = quaternion_from_euler(0, 0, syaw)
            s.pose.orientation.x = q[0]
            s.pose.orientation.y = q[1]
            s.pose.orientation.z = q[2]
            s.pose.orientation.w = q[3]

            rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(s)
            self.pub.publish(AckermannDrive())
        except Exception:
            pass

    # ============================
    # MAIN LOOP
    # ============================
    def run(self):
        if not self.initialized:
            return

        while not rospy.is_shutdown():
            if self.current_state is None:
                self.rate.sleep()
                continue

            t_now = float(rospy.get_time())

            # Online eval: compare last step prediction to current measurement
            self.try_eval_one_step(t_now, self.current_state)

            # Signed CTE log
            cte, _ = self.signed_cte(self.current_state)
            self.cte_log.append([t_now, cte])

            ref_traj = self.get_reference_trajectory()
            if ref_traj is None:
                self.rate.sleep()
                continue

            use_warm = self.prev_u_opt is not None
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

                # save prev command for eval record BEFORE updating memory
                prev_cmd_v = float(self.last_cmd_v)
                prev_cmd_s = float(self.last_cmd_steer)

                # Update command memory states (match acados state update)
                self.last_cmd_steer = cmd_s
                self.last_cmd_v = cmd_v

                # Filter outputs (what you actually send)
                filtered_steer = ALPHA_STEER * cmd_s + (1.0 - ALPHA_STEER) * self.last_pub_steer
                self.last_pub_steer = filtered_steer
                pub_steer = (filtered_steer / STEER_GAIN_COMP) + STEER_OFFSET

                filtered_v = ALPHA_V * cmd_v + (1.0 - ALPHA_V) * self.last_pub_v
                self.last_pub_v = filtered_v

                msg.speed = filtered_v
                msg.steering_angle = pub_steer

                # Store eval record for next step (use SAME u seen by model, not filtered)
                self._eval_prev = {
                    "t": t_now,
                    "state6": self.current_state.copy(),
                    "cmd_v": cmd_v,
                    "cmd_s": cmd_s,
                    "prev_cmd_v": prev_cmd_v,
                    "prev_cmd_s": prev_cmd_s,
                }

                if self.vis_pub.get_num_connections() > 0:
                    self.publish_markers(ref_traj, x_opt)
            else:
                msg.speed = 0.0
                msg.steering_angle = 0.0
                print("Acados Solver Failed -> Stopping")

                # still update eval record so the next step doesn't explode with stale
                self._eval_prev = None

            self.pub.publish(msg)

            # CSV row log
            self.log_data.append([
                t_now,
                cte,
                msg.steering_angle,
                msg.speed,
                self._steer_data,
                self._yaw_rate_data,
                float(self.current_state[3]),
                self.imu_data["ax"],
                self.imu_data["ay"],
                self.imu_data["az"],
                self.imu_data["wz"],
                self.last_cmd_v,
                self.last_cmd_steer,
            ])

            if res is not None:
                self.traj_log.append({"time": t_now, "ref": ref_traj, "pred": x_opt})

            self.rate.sleep()


if __name__ == "__main__":
    c = NeuralMPC()
    try:
        c.run()
    except rospy.ROSInterruptException:
        pass
