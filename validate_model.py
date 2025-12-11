#!/usr/bin/env python3
import numpy as np
import torch
import sys
import pickle

# --- Configuration ---
MODEL_PATH = 'gem_dynamics.pth'
SCALER_PATH = 'gem_scaler.pkl'
SCALER_ARRAY_PATH = 'gem_scaler_arrays.npz'
DT = 0.1 # Testing with base DT

class ModelValidator:
    def __init__(self):
        # 1. Load Scalers
        self.load_scalers()
        
        # 2. Load Weights
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        self.weights = {}
        for key, val in state_dict.items():
            self.weights[key] = val.cpu().numpy()
        print("Model weights loaded.")

    def load_scalers(self):
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = np.core
            
        try:
            with open(SCALER_PATH, 'rb') as f:
                scalers = pickle.load(f)
                self.sx_mean = np.asarray(scalers['x'].mean_, dtype=float)
                self.sx_scale = np.asarray(scalers['x'].scale_, dtype=float)
                self.sy_mean = np.asarray(scalers['y'].mean_, dtype=float)
                self.sy_scale = np.asarray(scalers['y'].scale_, dtype=float)
            print("Scalers loaded from pickle.")
        except Exception as e:
            print(f"Pickle load failed: {e}. Trying npz.")
            try:
                arrs = np.load(SCALER_ARRAY_PATH)
                self.sx_mean = arrs['x_mean']
                self.sx_scale = arrs['x_scale']
                self.sy_mean = arrs['y_mean']
                self.sy_scale = arrs['y_scale']
                print("Scalers loaded from npz.")
            except Exception as e2:
                print(f"Failed to load scalers: {e2}")
                sys.exit(1)

    def predict(self, v_actual, v_cmd, steer_cmd, steer_actual, dt=DT):
        # Input: [v_actual, v_cmd, steer_cmd, steer_actual, dt]
        inp = np.array([v_actual, v_cmd, steer_cmd, steer_actual, dt], dtype=float)
        inp_norm = (inp - self.sx_mean) / self.sx_scale

        # Forward Pass (Tanh activations)
        act = np.tanh
        h1 = act(self.weights['net.0.weight'].dot(inp_norm) + self.weights['net.0.bias'])
        h2 = act(self.weights['net.2.weight'].dot(h1) + self.weights['net.2.bias'])
        h3 = act(self.weights['net.4.weight'].dot(h2) + self.weights['net.4.bias'])
        out_norm = self.weights['net.6.weight'].dot(h3) + self.weights['net.6.bias']
        
        # Denormalize Output: [dx_body, dy_body, d_yaw, d_v]
        out_real = out_norm * self.sy_scale + self.sy_mean
        return out_real

    def expected_kinematic(self, v, steer, dt):
        """Calculates expected changes using Kinematic Bicycle Model"""
        L = 1.75 # Wheelbase
        # d_yaw = (v/L) * tan(steer) * dt
        d_yaw = (v / L) * np.tan(steer) * dt
        # dx_body = v * dt (approx for small angles)
        dx = v * dt
        dy = 0.0 # Standard bicycle model doesn't drift immediately
        return dx, dy, d_yaw

    def check(self, name, actual, expected, tol_sign=True, min_mag_ratio=0.1):
        """
        Checks if actual matches expected.
        tol_sign: If True, checks if signs match (for steering).
        min_mag_ratio: checks if magnitude is at least x% of expected.
        """
        passed = True
        reasons = []

        # Check Sign
        if tol_sign and expected != 0:
            if np.sign(actual) != np.sign(expected) and abs(actual) > 1e-4:
                passed = False
                reasons.append(f"WRONG SIGN (Got {actual:.4f}, Expected sign of {expected:.4f})")
        
        # Check Magnitude (Sensitivity)
        if expected != 0:
            ratio = abs(actual / expected)
            if ratio < min_mag_ratio:
                passed = False
                reasons.append(f"WEAK RESPONSE (Ratio {ratio:.2f} < {min_mag_ratio})")

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if not passed:
            for r in reasons:
                print(f"       -> {r}")
        return passed

    def run_tests(self):
        print("\n--- Running Sophisticated Validation (DT=0.1s) ---")
        print(f"Wheelbase: 1.75m. Kinematic baseline used for comparison.")
        
        dt = 0.1
        failures = 0

        # Test 1: Stopped
        print("\nTest 1: STOPPED")
        res = self.predict(0, 0, 0, 0, dt)
        if abs(res[0]) < 0.01 and abs(res[2]) < 0.001:
            print("[PASS] Stationary")
        else:
            print(f"[FAIL] Drifting while stopped! dx={res[0]:.3f}, d_yaw={res[2]:.3f}")
            failures += 1

        # Test 2: Straight
        print("\nTest 2: STRAIGHT (v=1.0, steer=0.0)")
        res = self.predict(1.0, 1.0, 0.0, 0.0, dt)
        k_dx, k_dy, k_yaw = self.expected_kinematic(1.0, 0.0, dt)
        # Check Speed
        if self.check("Longitudinal Motion", res[0], k_dx, tol_sign=True, min_mag_ratio=0.8):
            pass
        else:
            failures += 1
        # Check Yaw (Should be near zero)
        if abs(res[2]) < 0.005: 
             print(f"[PASS] Straight Line Stability (Yaw change {res[2]:.4f} is low)")
        else:
             print(f"[FAIL] Swerving while straight! d_yaw={res[2]:.4f}")
             failures += 1

        # Test 3: Hard Left
        steer = 0.5
        print(f"\nTest 3: HARD LEFT (v=1.0, steer={steer})")
        res = self.predict(1.0, 1.0, steer, steer, dt)
        k_dx, k_dy, k_yaw = self.expected_kinematic(1.0, steer, dt)
        print(f"       Model: d_yaw={res[2]:.4f} | Kinematic: {k_yaw:.4f}")
        
        if not self.check("Left Turn Yaw Response", res[2], k_yaw, min_mag_ratio=0.5):
            failures += 1

        # Test 4: Hard Right
        steer = -0.5
        print(f"\nTest 4: HARD RIGHT (v=1.0, steer={steer})")
        res = self.predict(1.0, 1.0, steer, steer, dt)
        _, _, k_yaw = self.expected_kinematic(1.0, steer, dt)
        print(f"       Model: d_yaw={res[2]:.4f} | Kinematic: {k_yaw:.4f}")
        
        if not self.check("Right Turn Yaw Response", res[2], k_yaw, min_mag_ratio=0.5):
            failures += 1

        print("\n------------------------------------------------")
        if failures == 0:
            print("OVERALL RESULT: PASS (Model looks plausible)")
        else:
            print(f"OVERALL RESULT: FAIL ({failures} failures)")
            print("Recommendation: The model is not learning the dynamics correctly.")
            print("Possible causes: Poor data coverage, wrong normalization, or too much regularization.")

if __name__ == "__main__":
    validator = ModelValidator()
    validator.run_tests()
