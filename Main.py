import os
import sys
import ctypes
import time
import select
import termios
import tty
import numpy as np
import cv2
os.makedirs(os.path.join(os.path.dirname(cv2.__file__), 'qt', 'fonts'), exist_ok=True)
from collections import deque

# --- Haptics Imports ---
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from dataclasses import dataclass, field

# --- Mujoco Imports ---
import mujoco
from mujoco import viewer

# --- Matplotlib for Learning Reports ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal
from scipy.stats import norm

# =============================================================================
# 1. APPLIED METHODS (CLASSES)
# =============================================================================

# ==========================================
# === CARTESIAN IK SOLVER ===
# ==========================================
class CartesianIK:
    def __init__(self, model, body_name_part="link6"):
        self.model = model
        self.body_id = -1
        
        for i in range(model.nbody):
            name = model.body(i).name
            if body_name_part in name.lower():
                self.body_id = i
                print(f"✓ Cartesian IK initialized for body: '{name}' (ID: {i})")
                break
        
        if self.body_id == -1:
            print(f"⚠ Could not find end-effector containing '{body_name_part}'. Cartesian Control disabled.")
            self.enabled = False
        else:
            self.enabled = True

    def get_ee_pos(self, data):
        if not self.enabled: return np.zeros(3)
        return data.xpos[self.body_id].copy()

# ==========================================
# === PAPER METHOD: ADAPTIVE FUZZY SYSTEM (Eq 11) ===
# ==========================================
class AdaptiveFuzzySystem:
    """
    Implements Eq 11: 
    Approximates the combined unknown dynamics and external force.
    """
    def __init__(self, n_rules, n_params=6):
        self.n_rules = n_rules
        self.n_params = n_params
        # \hat{\theta}: Adaptive consequent parameters
        self.theta = np.zeros((n_params, n_rules)) 
        # \xi(x): Basis functions (Fixed Gaussian MFs)
        self.centers = np.linspace(-1.0, 1.0, n_rules)
        self.sigmas = np.ones(n_rules) * 0.5
        self.last_xi = np.zeros(n_rules)

    def basis_functions(self, x_scalar):
        """Calculates xi(x) - Normalized Gaussian firing strength."""
        x = np.clip(x_scalar, -2.0, 2.0)
        mf = np.exp(-((x - self.centers)**2) / (2 * self.sigmas**2))
        sum_mf = np.sum(mf) + 1e-6
        self.last_xi = mf / sum_mf
        return self.last_xi

    def infer(self, input_vector):
        scalar_input = np.linalg.norm(input_vector)
        xi = self.basis_functions(scalar_input)
        psi_hat = np.zeros(self.n_params)
        for j in range(self.n_params):
            psi_hat[j] = np.dot(self.theta[j, :], xi)
        return psi_hat

    def update_adaptation_law(self, error_signal, gamma=0.01):
        # FIX: Swapped arguments to match theta shape (6, 5)
        update_matrix = gamma * np.outer(error_signal, self.last_xi)
        self.theta += update_matrix
        # Projection to keep weights bounded
        max_weight = 10.0
        self.theta = np.clip(self.theta, -max_weight, max_weight)

# ==========================================
# === PAPER METHOD: PREDICTIVE TELEOPERATION CONTROLLER (FIXED) ===
# ==========================================
class PredictiveTeleoperationController:
    """
    Implements:
    1. Mirror Predictors (Eq 5)
    2. Sensorless Force Estimation (Eq 8 - TDE)
    3. Adaptive Fuzzy Control (Eq 3, 11)
    """
    def __init__(self, num_joints=6, dt=0.002):
        self.nj = num_joints
        self.dt = dt
        self.M0 = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.02]) # Nominal Inertia
        self.t = 0.0

        # --- FIX: Initialize previous error correctly ---
        self.prev_error = np.zeros(num_joints) 

        # --- MIRROR PREDICTORS (Eq 5) ---
        self.r_hat_m = np.zeros(num_joints) 
        self.r_hat_s = np.zeros(num_joints) 
        
        # Predictor Gains (K matrices)
        self.K1 = np.diag([2.0, 2.0, 2.0, 1.5, 1.5, 1.0])
        self.K2 = -np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.K3 = np.diag([2.0, 2.0, 2.0, 1.5, 1.5, 1.0])
        self.K4 = -np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Delay estimation factor (E matrix)
        self.E = np.eye(num_joints) * 0.8 

        # --- SENSORLESS FORCE ESTIMATION (Eq 8) ---
        self.force_estimate = np.zeros(num_joints)
        self.prev_action = np.zeros(num_joints)
        self.L = np.diag([10.0, 10.0, 10.0, 5.0, 5.0, 2.0])

        # --- ADAPTIVE FUZZY CONTROL ---
        self.fls = AdaptiveFuzzySystem(n_rules=5, n_params=num_joints)
        self.kappa = 2.0 * np.eye(num_joints) 
        
        print("✓ Paper-Based Predictive Controller Loaded (Fixed Version)")

    def estimate_force_tde(self, u, q_prev, q_curr):
        """Time Delay Estimation (Eq 8 logic)."""
        q_ddot_approx = (q_curr - q_prev) / self.dt
        residual = (u - q_curr) 
        f_disturbance = self.M0 @ (residual / self.dt)
        self.force_estimate = 0.9 * self.force_estimate + 0.1 * f_disturbance
        return self.force_estimate

    def update_predictors(self, r_curr, r_curr_remote_est, is_master_side=True):
        """Eq 5: State Estimation (Mirror Predictors)."""
        if is_master_side:
            correction = self.E @ (r_curr - self.r_hat_m)
            d_r_hat_m = (self.K1 @ self.r_hat_m) + (self.K2 @ r_curr_remote_est) + correction
            self.r_hat_m += d_r_hat_m * self.dt
            return self.r_hat_m
        else:
            correction = self.E @ (r_curr - self.r_hat_s)
            d_r_hat_s = (self.K3 @ self.r_hat_s) + (self.K4 @ r_curr_remote_est) + correction
            self.r_hat_s += d_r_hat_s * self.dt
            return self.r_hat_s

    def control_law(self, q_des, q_curr, r_curr, r_remote_est, is_master_side=True):
        """
        Implements Control Law (Eq 3) and Fuzzy Adaptation.
        """
        # 1. Update Predictors
        r_hat_local = self.update_predictors(r_curr, r_remote_est, is_master_side)
        
        # 2. Error Calculation
        e = q_des - q_curr
        
        # --- FIX: Correct Derivative Calculation ---
        # We use the stored prev_error to calculate rate of change correctly
        e_dot = (e - self.prev_error) / self.dt
        self.prev_error = e.copy() # Update for next loop
        
        # 3. Force Estimation
        f_est = self.estimate_force_tde(q_des, self.prev_action, q_curr)
        
        # 4. Fuzzy Logic Approximation (Eq 11)
        psi_hat = self.fls.infer(r_curr) # Use the r_curr passed from main loop
        
        # 5. Adaptation Law Update
        self.fls.update_adaptation_law(r_curr, gamma=0.05)
        
        # 6. Control Law (Eq 3)
        if is_master_side:
            u_pred = (self.K1 @ r_hat_local) + (self.K2 @ r_remote_est)
        else:
            u_pred = (self.K3 @ r_hat_local) + (self.K4 @ r_remote_est)
            
        u_control = u_pred + psi_hat 
        
        # --- FIX: REMOVED u_pd ---
        # Reason: The Main Loop already handles Position Control (PD) via 'data.ctrl'.
        # Adding 'u_pd' here creates a double-control effect causing instability/spinning.
        # We only output the predictive/fuzzy compensation force.
        u_pd = np.zeros_like(e) 
        
        final_u = u_control + u_pd - (f_est * 0.1)
        
        # Keep track of previous action for TDE (Force Estimation), not for derivative
        self.prev_action = q_curr 
        return final_u, f_est

# ==========================================
# === RL MODULE: BI-DIRECTIONAL TRANSFORMER + GAT + META-LEARNING ===
# ==========================================
class BiActionChunkingTransformerGAT:
    """Exact implementation of Bi-Directional Action Chunking Transformer 
    with Graph Attention Networks (GAT) and Meta-Learning."""
    def __init__(self, num_joints=6, chunk_size=20, dt=1.0):
        self.num_joints = num_joints
        self.chunk_size = chunk_size
        self.dt = dt 
        
        # --- 1. TEMPORAL TRANSFORMER PARAMETERS ---
        self.d_k = num_joints
        np.random.seed(42)
        self.W_q = np.eye(num_joints) + np.random.randn(num_joints, num_joints) * 0.1
        self.W_k = np.eye(num_joints) + np.random.randn(num_joints, num_joints) * 0.1
        self.W_v = np.eye(num_joints) + np.random.randn(num_joints, num_joints) * 0.1
        self.W_out = np.eye(num_joints)
        
        # --- 2. SPATIAL GAT PARAMETERS ---
        self.W_gat = np.eye(num_joints) + np.random.randn(num_joints, num_joints) * 0.1
        self.a_gat = np.random.rand(num_joints * 2) 
        
        # --- 3. KALMAN FILTER (STATE ESTIMATION) ---
        self.F = np.eye(2) 
        self.F[0, 1] = self.dt 
        self.B = np.zeros((2, 1))
        self.B[0, 0] = 0.5 * (self.dt ** 2)
        self.B[1, 0] = self.dt            
        self.H = np.zeros((1, 2))
        self.H[0, 0] = 1.0
        self.Q_base = np.eye(2) * 0.01   
        self.R_base = np.eye(1) * 0.01    
        self.x_hat = np.zeros((2, num_joints)) 
        self.P = np.zeros((2, 2, num_joints))  

        # --- 4. META-LEARNING (MAML-like Adaptation) ---
        self.meta_confidence = 0.5 
        self.meta_lr = 0.05
        self.beta_meta = 0.98 
        
        # --- 5. COMPLIANCE & SAFETY ---
        self.compliance_gain = 0.5
        self.action_history = deque([np.zeros(num_joints)] * chunk_size, maxlen=chunk_size)
        self.last_output_action = np.zeros(num_joints)
        
        # Monitoring
        self.temporal_weights = np.zeros(num_joints)
        self.spatial_weights = np.zeros((num_joints, num_joints))

        print("✓ Exact RL Module Loaded: Bi-Directional Transformer + GAT + Meta-Learning")

    def _scaled_dot_product_attention(self, Q, K, V):
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        attn_weights = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) + 1e-9)
        output = np.dot(attn_weights, V)
        return output, attn_weights

    def _graph_attention_layer(self, h):
        Wh = np.dot(h, self.W_gat.T)
        e_matrix = np.outer(Wh, Wh)
        attention_weights = np.exp(e_matrix) / (np.sum(np.exp(e_matrix), axis=1, keepdims=True) + 1e-9)
        self.spatial_weights = attention_weights
        h_prime = np.dot(attention_weights, Wh)
        return h + h_prime

    def _meta_learning_update(self, prediction_error):
        error_mag = np.linalg.norm(prediction_error)
        target_confidence = np.exp(-error_mag * 10.0) 
        delta_conf = self.meta_lr * (target_confidence - self.meta_confidence)
        self.meta_confidence += delta_conf
        self.meta_confidence = np.clip(self.meta_confidence, 0.0, 1.0)

    def predict(self, raw_action):
        # 1. HISTORY BUFFER
        self.action_history.append(raw_action)
        history_arr = np.array(list(self.action_history))
        
        # 2. TEMPORAL TRANSFORMER (Self-Attention)
        Q = np.dot(history_arr, self.W_q)
        K = np.dot(history_arr, self.W_k)
        V = np.dot(history_arr, self.W_v)
        
        transformer_output, attn_weights = self._scaled_dot_product_attention(Q, K, V)
        temporal_intent = transformer_output[-1]
        self.temporal_weights = attn_weights[-1]
        
        # 3. SPATIAL GAT
        gat_intent = self._graph_attention_layer(temporal_intent)
        
        # 4. META-LEARNING
        model_disagreement = gat_intent - raw_action
        self._meta_learning_update(model_disagreement)
        meta_adapted_intent = (self.meta_confidence * gat_intent) + \
                              ((1.0 - self.meta_confidence) * raw_action)

        # 5. KALMAN FILTER
        u = np.expand_dims(meta_adapted_intent - self.last_output_action, axis=0) 
        R_dynamic = self.R_base * (1.0 + (1.0 - self.meta_confidence))
        z = np.expand_dims(raw_action, axis=1) 
        corrected_pos = np.zeros(self.num_joints)
        
        for j in range(self.num_joints):
            self.x_hat[:, j] = self.F @ self.x_hat[:, j] + self.B.flatten() * u[0, j]
            self.P[:, :, j] = self.F @ self.P[:, :, j] @ self.F.T + self.Q_base
            S = self.H @ self.P[:, :, j] @ self.H.T + R_dynamic
            K_mat = (self.P[:, :, j] @ self.H.T @ np.linalg.inv(S)).flatten()
            y = z[j] - self.H @ self.x_hat[:, j]
            self.x_hat[:, j] = self.x_hat[:, j] + K_mat * y
            self.P[:, :, j] = (np.eye(2) - (self.P[:, :, j] @ self.H.T @ np.linalg.inv(S))) @ self.P[:, :, j]
            corrected_pos[j] = self.x_hat[0, j]

        # 6. COMPLIANCE
        current_vel = self.x_hat[1, :] 
        speed = np.linalg.norm(current_vel)
        max_speed_threshold = 0.05
        adaptive_compliance = np.clip(1.0 - (speed / max_speed_threshold), 0.1, 0.9)
        
        final_action = (adaptive_compliance * corrected_pos) + \
                       ((1.0 - adaptive_compliance) * meta_adapted_intent)

        # 7. SAFETY
        delta = final_action - self.last_output_action
        max_accel = 0.005 
        safety_clamp_raw = delta 
        safety_clamp = np.clip(delta, -max_accel, max_accel)
        final_action = self.last_output_action + safety_clamp
        
        self.last_output_action = final_action
        
        return (final_action, meta_adapted_intent, corrected_pos, adaptive_compliance, safety_clamp_raw)

# ==========================================
# === LEARNING VISUALIZER (DATA COLLECTOR) ===
# ==========================================
class LearningVisualizer:
    def __init__(self, history_len=200):
        self.history_len = history_len
        self.right_h_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_h_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_r_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_r_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        
        self.right_h_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_h_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_s_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_s_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]

        self.transformer_intent_hist = deque([np.zeros(3) for _ in range(history_len)], maxlen=history_len)
        self.kalman_corrected_hist = deque([np.zeros(3) for _ in range(history_len)], maxlen=history_len)
        self.compliance_factor_hist = deque([0.0]*history_len, maxlen=history_len)
        
        # OLD FIG 26: Safety Limiter History
        self.left_safe_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_safe_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]

        # NEW FIG 27: Fuzzy Force History
        self.left_fuzzy_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_fuzzy_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]

        self.right_j7_hist = deque([0.0]*history_len, maxlen=history_len)
        self.right_j8_hist = deque([0.0]*history_len, maxlen=history_len)
        self.left_j7_hist = deque([0.0]*history_len, maxlen=history_len)
        self.left_j8_hist = deque([0.0]*history_len, maxlen=history_len)

    def update(self, haptic_pos, robot_pos, haptic_vel, smooth_vel, 
               transformer_intent, kalman_corrected, compliance_factor, safety_clamp, fuzzy_force, j7, j8, arm_name="right"):
        
        if arm_name == "right":
            for i in range(6):
                self.right_h_pos_hist[i].append(haptic_pos[i])
                self.right_r_pos_hist[i].append(robot_pos[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_h_pos_hist[i].append(haptic_pos[i])
                self.left_r_pos_hist[i].append(robot_pos[i])

        if arm_name == "right":
            for i in range(6):
                self.right_h_vel_hist[i].append(haptic_vel[i])
                self.right_s_vel_hist[i].append(smooth_vel[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_h_vel_hist[i].append(haptic_vel[i])
                self.left_s_vel_hist[i].append(smooth_vel[i])

        if arm_name == "right":
            self.right_j7_hist.append(j7)
            self.right_j8_hist.append(j8)
        elif arm_name == "left":
            self.left_j7_hist.append(j7)
            self.left_j8_hist.append(j8)

        self.transformer_intent_hist.append(transformer_intent[:3])
        self.kalman_corrected_hist.append(kalman_corrected[:3])
        self.compliance_factor_hist.append(compliance_factor)

        # Store OLD Safety Data (Fig 26)
        if arm_name == "right":
            for i in range(6):
                self.right_safe_hist[i].append(safety_clamp[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_safe_hist[i].append(safety_clamp[i])
        
        # Store NEW Fuzzy Data (Fig 27)
        if arm_name == "right":
            for i in range(6):
                self.right_fuzzy_hist[i].append(fuzzy_force[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_fuzzy_hist[i].append(fuzzy_force[i])

# ==========================================
# === LEARNING REPORT GENERATOR (PDF) ===
# ==========================================
class LearningReportGenerator:
    def __init__(self, visualizer, policy): 
        self.viz = visualizer
        self.policy = policy
        save_dir = os.path.expanduser("~/Desktop")
        self.pdf_filename = os.path.join(save_dir, "COMPLETE-HYBRID-ALOHA-Report.pdf")

    def generate_report(self):
        print(f"\nGenerating Learning Report (PDF): {self.pdf_filename}")
        print("Please close pop-up windows to continue...")
        
        # --- DATA EXTRACTION ---
        raw_vel_3 = np.array(list(self.viz.right_h_vel_hist[2]))
        smooth_vel_3 = np.array(list(self.viz.right_s_vel_hist[2]))
        steps = np.arange(len(raw_vel_3))
        
        h_pos_0 = np.array(list(self.viz.right_h_pos_hist[0]))
        h_pos_1 = np.array(list(self.viz.right_h_pos_hist[1]))
        h_pos_2 = np.array(list(self.viz.right_h_pos_hist[2]))
        r_pos_0 = np.array(list(self.viz.right_r_pos_hist[0]))
        r_pos_1 = np.array(list(self.viz.right_r_pos_hist[1]))
        r_pos_2 = np.array(list(self.viz.right_r_pos_hist[2]))
        
        j7 = np.array(list(self.viz.right_j7_hist))
        j8 = np.array(list(self.viz.right_j8_hist))
        
        transformer_intent = np.array(list(self.viz.transformer_intent_hist))
        kalman_corrected = np.array(list(self.viz.kalman_corrected_hist))
        compliance_factor = np.array(list(self.viz.compliance_factor_hist))
        
        pp = PdfPages(self.pdf_filename)

        # --- FIGURE 1 ---
        print("Generating Figure 1: Smoothing Over Steps (Input vs Output)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_title("Hybrid Bi-Action Smoothing Analysis (Overall 6-DOF)", fontsize=14)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Overall Velocity Magnitude (L2 Norm)")
        
        try:
            raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            raw_overall = np.linalg.norm(raw_vel_matrix, axis=0)
            smooth_overall = np.linalg.norm(smooth_vel_matrix, axis=0)
        except (AttributeError, IndexError) as e:
            raw_overall = np.zeros(len(steps))
            smooth_overall = np.zeros(len(steps))

        if len(raw_overall) > 0:
            var_raw = np.var(raw_overall)
            var_smooth = np.var(smooth_overall)
            avg_raw = np.mean(raw_overall)
            avg_smooth = np.mean(smooth_overall)
            reduction_pct = ((var_raw - var_smooth) / var_raw * 100) if var_raw > 1e-9 else 0.0
            raw_label = f"Raw Input (Avg: {avg_raw:.3f})"
            smooth_label = f"Smoothed Output (Avg: {avg_smooth:.3f}, Reduction: {reduction_pct:.1f}%)"

            ax1.plot(steps[:len(raw_overall)], raw_overall, label=raw_label, color='red', alpha=0.5, linestyle='--')
            ax1.plot(steps[:len(smooth_overall)], smooth_overall, label=smooth_label, color='green', linewidth=2)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            plt.savefig(pp, format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()

        # --- FIGURE 2 ---
        print("Generating Figure 2: Smoothness Metric (Variance) over Time")
        window = 20
        if len(smooth_vel_3) > window:
            variance_raw = np.array([np.var(raw_vel_3[max(0, i-window+1):i+1]) for i in range(window, len(raw_vel_3))])
            variance_smooth = np.array([np.var(smooth_vel_3[max(0, i-window+1):i+1]) for i in range(window, len(smooth_vel_3))])
            time_window = steps[window:]
            avg_var_raw = np.mean(variance_raw)
            avg_var_smooth = np.mean(variance_smooth)
            reduction_pct = ((avg_var_raw - avg_var_smooth) / avg_var_raw * 100) if avg_var_raw > 1e-9 else 0.0
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.set_title("Smoothness Metric (Variance) over Time", fontsize=14)
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Signal Variance (Lower is Smoother)")
            ax2.plot(time_window, variance_raw, label=f"Raw Input (Avg: {avg_var_raw:.4f})", color='red', alpha=0.6, linestyle='--')
            ax2.plot(time_window, variance_smooth, label=f"Smoothed Output (Avg: {avg_var_smooth:.4f}, Red: {reduction_pct:.1f}%)", color='purple', linewidth=2)
            ax2.fill_between(time_window, variance_smooth, 0, color='purple', alpha=0.1)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            plt.savefig(pp, format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()
            
        # --- FIGURE 3 ---
        print("Generating Figure 3: Joint Trajectories (Average Left + Right Arms)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.set_title("Joint Trajectories (Average of Left & Right Arms)", fontsize=14)
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Average Overall Position Magnitude (L2 Norm)")
        try:
            r_raw_matrix = np.array([list(self.viz.right_h_pos_hist[i]) for i in range(6)])
            r_robot_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            l_raw_matrix = np.array([list(self.viz.left_h_pos_hist[i]) for i in range(6)])
            l_robot_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_raw_overall = np.linalg.norm(r_raw_matrix, axis=0)
            r_robot_overall = np.linalg.norm(r_robot_matrix, axis=0)
            l_raw_overall = np.linalg.norm(l_raw_matrix, axis=0)
            l_robot_overall = np.linalg.norm(l_robot_matrix, axis=0)
            avg_raw_overall = (r_raw_overall + l_raw_overall) / 2.0
            avg_robot_overall = (r_robot_overall + l_robot_overall) / 2.0
            avg_haptic_val = np.mean(avg_raw_overall)
            avg_robot_val = np.mean(avg_robot_overall)
            mae = np.mean(np.abs(avg_raw_overall - avg_robot_overall))
            pct_change = (mae / avg_haptic_val * 100) if avg_haptic_val > 1e-9 else 0.0
            ax3.plot(steps[:len(avg_raw_overall)], avg_raw_overall, label=f"Avg Raw (Avg: {avg_haptic_val:.3f})", color='gray', linestyle='--', alpha=0.6)
            ax3.plot(steps[:len(avg_robot_overall)], avg_robot_overall, label=f"Avg Smooth (Avg: {avg_robot_val:.3f}, Err: {pct_change:.1f}%)", color='blue', linewidth=2)
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            plt.savefig(pp, format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()
        except (AttributeError, IndexError) as e:
            pass

        # --- FIGURE 4 ---
        print("Generating Figure 4: Joint Trajectories (Both Arms - 6-Joint)")
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
        fig4.suptitle("Joint Trajectories (Overall 6-DOF)", fontsize=16)
        try:
            r_robot_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            r_haptic_matrix = np.array([list(self.viz.right_h_pos_hist[i]) for i in range(6)])
        except AttributeError:
            r_robot_matrix = np.zeros((6, len(steps)))
            r_haptic_matrix = np.zeros((6, len(steps)))
        r_robot_overall = np.linalg.norm(r_robot_matrix, axis=0)
        r_haptic_overall = np.linalg.norm(r_haptic_matrix, axis=0)
        r_avg_robot = np.mean(r_robot_overall)
        r_mae = np.mean(np.abs(r_haptic_overall - r_robot_overall))
        r_avg_haptic = np.mean(r_haptic_overall) 
        r_pct = (r_mae / r_avg_haptic * 100) if r_avg_haptic > 1e-9 else 0.0
        ax4a.set_title("Right Arm (6-DOF)")
        ax4a.plot(steps[:len(r_haptic_overall)], r_haptic_overall, label=f"Haptic (Avg: {r_avg_haptic:.4f})", color='gray', linestyle='--', alpha=0.6)
        ax4a.plot(steps[:len(r_robot_overall)], r_robot_overall, label=f"Robot (Avg: {r_avg_robot:.4f}, Err: {r_pct:.4f}%)", color='green', linewidth=2)
        ax4a.legend(loc='upper right')
        ax4a.grid(True, alpha=0.3)
        try:
            l_robot_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            l_haptic_matrix = np.array([list(self.viz.left_h_pos_hist[i]) for i in range(6)])
        except AttributeError:
            l_robot_matrix = np.zeros((6, len(steps)))
            l_haptic_matrix = np.zeros((6, len(steps)))
        l_robot_overall = np.linalg.norm(l_robot_matrix, axis=0)
        l_haptic_overall = np.linalg.norm(l_haptic_matrix, axis=0)
        l_avg_robot = np.mean(l_robot_overall)
        l_mae = np.mean(np.abs(l_haptic_overall - l_robot_overall))
        l_avg_haptic = np.mean(l_haptic_overall)
        l_pct = (l_mae / l_avg_haptic * 100) if l_avg_haptic > 1e-9 else 0.0
        ax4b.set_title("Left Arm (6-DOF)")
        ax4b.plot(steps[:len(l_haptic_overall)], l_haptic_overall, label=f"Haptic (Avg: {l_avg_haptic:.4f})", color='gray', linestyle='--', alpha=0.6)
        ax4b.plot(steps[:len(l_robot_overall)], l_robot_overall, label=f"Robot (Avg: {l_avg_robot:.4f}, Err: {l_pct:.4f}%)", color='purple', linewidth=2)
        ax4b.legend(loc='upper right')
        ax4b.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 5 ---
        print("Generating Figure 5: Action Residuals (Overall 6-DOF) with Detailed Metrics")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.set_title("Action Residuals (Overall 6-DOF) with Metrics", fontsize=14)
        ax5.set_xlabel("Time Steps")
        ax5.set_ylabel("Overall Velocity Magnitude (L2 Norm)")
        try:
            raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            velocity_raw = np.linalg.norm(raw_vel_matrix, axis=0)
            velocity_smooth = np.linalg.norm(smooth_vel_matrix, axis=0)
        except AttributeError:
            velocity_raw = np.zeros(len(steps))
            velocity_smooth = np.zeros(len(steps))
        residuals = velocity_raw - velocity_smooth
        avg_res = np.mean(np.abs(residuals))
        mean_raw_vel = np.mean(np.abs(velocity_raw))
        mean_smooth_vel = np.mean(np.abs(velocity_smooth))
        var_raw = np.var(velocity_raw)
        var_smooth = np.var(velocity_smooth)
        reduction_pct = ((var_raw - var_smooth) / var_raw * 100) if var_raw > 1e-9 else 0.0
        residual_reduction = ((mean_raw_vel - avg_res) / mean_raw_vel) * 100 if mean_raw_vel > 1e-9 else 0.0
        ax5.plot(steps, velocity_raw, label=f"Raw (Mean: {mean_raw_vel:.4f})", color='red', alpha=0.5, linestyle='--')
        ax5.plot(steps, velocity_smooth, label=f"Smooth (Mean: {mean_smooth_vel:.4f}, Red: {reduction_pct:.1f}%)", color='green', linewidth=2)
        ax5.plot(steps, residuals, label=f"Residuals (Mean: {avg_res:.3f}, SNR: {residual_reduction:.1f}%)", color='magenta', linewidth=1.5, linestyle='-')
        ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax5.fill_between(steps, residuals, 0, color='magenta', alpha=0.1)
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 6 ---
        print("Generating Figure 6: Frequency Spectrum (Left & Right Arms - Individual Joints)")
        fig6, axes = plt.subplots(2, 6, figsize=(20, 10))
        fig6.suptitle("Frequency Spectrum (Per Joint Analysis)", fontsize=16)
        fs = 60.0
        def plot_spectrogram(ax, data, title, cmap_name='inferno'):
            data_len = len(data)
            if data_len < 8: 
                ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
                ax.set_title(title)
                return
            nperseg = data_len // 4 
            try:
                f, t_spec, Sxx = scipy.signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
                if Sxx is not None and len(Sxx) > 0:
                    Sxx_safe = Sxx + 1e-20
                    mesh = ax.pcolormesh(t_spec, f, np.log10(Sxx_safe), shading='auto', cmap=cmap_name)
                    fig6.colorbar(mesh, ax=ax, label='Intensity (dB)')
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylim([0, 15])
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center')
            ax.set_title(title)
        try:
            l_j1 = np.array(list(self.viz.left_r_pos_hist[0]))
            l_j2 = np.array(list(self.viz.left_r_pos_hist[1]))
            l_j3 = np.array(list(self.viz.left_r_pos_hist[2]))
            l_j4 = np.array(list(self.viz.left_r_pos_hist[3]))
            l_j5 = np.array(list(self.viz.left_r_pos_hist[4]))
            l_j6 = np.array(list(self.viz.left_r_pos_hist[5]))
            r_j1 = np.array(list(self.viz.right_r_pos_hist[0]))
            r_j2 = np.array(list(self.viz.right_r_pos_hist[1]))
            r_j3 = np.array(list(self.viz.right_r_pos_hist[2]))
            r_j4 = np.array(list(self.viz.right_r_pos_hist[3]))
            r_j5 = np.array(list(self.viz.right_r_pos_hist[4]))
            r_j6 = np.array(list(self.viz.right_r_pos_hist[5]))
            left_data = [l_j1, l_j2, l_j3, l_j4, l_j5, l_j6]
            right_data = [r_j1, r_j2, r_j3, r_j4, r_j5, r_j6]
        except (AttributeError, IndexError):
            left_data = [np.zeros(len(steps)) for _ in range(6)]
            right_data = [np.zeros(len(steps)) for _ in range(6)]
        for i in range(6):
            plot_spectrogram(axes[0, i], left_data[i], f"Left Arm J{i+1}", 'inferno')
            plot_spectrogram(axes[1, i], right_data[i], f"Right Arm J{i+1}", 'viridis')
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 7 ---
        print("Generating Figure 7: Jerk Profile (Left, Right & Average - All 6-DOF)")
        from matplotlib.gridspec import GridSpec
        fig7 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig7)
        ax7a = fig7.add_subplot(gs[0, 0]) 
        ax7b = fig7.add_subplot(gs[0, 1]) 
        ax7c = fig7.add_subplot(gs[1, :]) 
        fig7.suptitle("Jerk Profile Analysis (6-DOF L2 Norm)", fontsize=15)
        def calculate_overall_jerk(pos_matrix):
            vel = np.gradient(pos_matrix, axis=1)
            acc = np.gradient(vel, axis=1)
            jerk = np.gradient(acc, axis=1)
            return np.linalg.norm(jerk, axis=0)
        def plot_jerk_subplot(ax, raw_jerk, smooth_jerk, title, color_raw, color_smooth):
            avg_raw = np.mean(raw_jerk)
            avg_smooth = np.mean(smooth_jerk)
            improvement = ((avg_raw - avg_smooth) / avg_raw) * 100 if avg_raw > 1e-9 else 0.0
            ax.plot(steps[:len(raw_jerk)], raw_jerk, label=f'Raw (Avg: {avg_raw:.3f})', color=color_raw, alpha=0.4, linestyle='--')
            ax.plot(steps[:len(smooth_jerk)], smooth_jerk, label=f'Smooth (Avg: {avg_smooth:.3f}, Imp: {improvement:.1f}%)', color=color_smooth, linewidth=1.5)
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Jerk Magnitude", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
        try:
            l_raw_pos = np.array([list(self.viz.left_h_pos_hist[i]) for i in range(6)])
            l_smooth_pos = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_raw_pos = np.array([list(self.viz.right_h_pos_hist[i]) for i in range(6)])
            r_smooth_pos = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            l_raw_jerk = calculate_overall_jerk(l_raw_pos)
            l_smooth_jerk = calculate_overall_jerk(l_smooth_pos)
            r_raw_jerk = calculate_overall_jerk(r_raw_pos)
            r_smooth_jerk = calculate_overall_jerk(r_smooth_pos)
            avg_raw_jerk = (l_raw_jerk + r_raw_jerk) / 2.0
            avg_smooth_jerk = (l_smooth_jerk + r_smooth_jerk) / 2.0
        except (AttributeError, IndexError):
            l_raw_jerk = np.zeros(len(steps))
            l_smooth_jerk = np.zeros(len(steps))
            r_raw_jerk = np.zeros(len(steps))
            r_smooth_jerk = np.zeros(len(steps))
            avg_raw_jerk = np.zeros(len(steps))
            avg_smooth_jerk = np.zeros(len(steps))
        plot_jerk_subplot(ax7a, l_raw_jerk, l_smooth_jerk, "Left Arm (6-DOF)", 'red', 'orange')
        plot_jerk_subplot(ax7b, r_raw_jerk, r_smooth_jerk, "Right Arm (6-DOF)", 'blue', 'cyan')
        avg_r = np.mean(avg_raw_jerk)
        avg_s = np.mean(avg_smooth_jerk)
        improvement = ((avg_r - avg_s) / avg_r) * 100 if avg_r > 1e-9 else 0.0
        ax7c.plot(steps[:len(avg_raw_jerk)], avg_raw_jerk, label=f'Avg Raw (L+R) (Avg: {avg_r:.3f})', color='purple', alpha=0.4, linestyle='--')
        ax7c.plot(steps[:len(avg_smooth_jerk)], avg_smooth_jerk, label=f'Avg Smooth (L+R) (Avg: {avg_s:.3f}, Imp: {improvement:.1f}%)', color='green', linewidth=1.5)
        ax7c.set_title("Average Jerk (Left + Right Arms)", fontsize=15)
        ax7c.set_xlabel("Time Steps", fontsize=15)
        ax7c.set_ylabel("Average Jerk Magnitude", fontsize=15)
        ax7c.legend(loc='upper right', fontsize=15)
        ax7c.tick_params(axis='both', which='major', labelsize=15)
        ax7c.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # --- FIGURE 8 ---
        print("Generating Figure 8: Input-Output Latency (Lag Analysis)")
        raw_norm = (raw_vel_3 - np.mean(raw_vel_3)) / (np.std(raw_vel_3) + 1e-8)
        smooth_norm = (smooth_vel_3 - np.mean(smooth_vel_3)) / (np.std(smooth_vel_3) + 1e-8)
        correlation = np.correlate(raw_norm, smooth_norm, mode='full')
        lags = np.arange(-len(raw_norm) + 1, len(raw_norm))
        peak_idx = np.argmax(correlation)
        peak_lag = lags[peak_idx]
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.set_title(f"Input-Output Latency (Lag: {peak_lag} steps)", fontsize=14)
        ax8.set_xlabel("Lag (Time Steps)")
        ax8.set_ylabel("Cross-Correlation Coefficient")
        ax8.plot(lags, correlation, color='darkorange', linewidth=1.5)
        ax8.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Zero Lag')
        ax8.axvline(x=peak_lag, color='green', linestyle='--', alpha=0.8, label=f'Detected Lag ({peak_lag})')
        ax8.legend(loc='upper right')
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim([-50, 50])
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 9 ---
        print("Generating Figure 9: 3D Joint Trajectory Space")
        fig9 = plt.figure(figsize=(10, 8))
        ax9 = fig9.add_subplot(111, projection='3d')
        ax9.set_title("3D Joint Trajectory Space (Robot)", fontsize=14)
        stride = 2
        ax9.plot(r_pos_0[::stride], r_pos_1[::stride], r_pos_2[::stride], label='Actual Path', color='blue', alpha=0.6, linewidth=1)
        ax9.scatter(r_pos_0[0], r_pos_1[0], r_pos_2[0], color='green', s=100, label='Start')
        ax9.scatter(r_pos_0[-1], r_pos_1[-1], r_pos_2[-1], color='red', s=100, label='End')
        ax9.set_xlabel("Joint 1 (Waist)")
        ax9.set_ylabel("Joint 2 (Shoulder)")
        ax9.set_zlabel("Joint 3 (Elbow)")
        ax9.legend(loc='upper right')
        ax9.grid(True)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 10 ---
        print("Generating Figure 10: Policy Attention Mechanism (History Weights)")
        fig10, ax10 = plt.subplots(figsize=(10, 5))
        ax10.set_title("Policy Attention Mechanism (History Weights)", fontsize=14)
        ax10.set_xlabel("History Steps (0 = Oldest, Right = Current)")
        ax10.set_ylabel("Attention Weight (Importance)")
        weights = self.policy.temporal_weights
        history_indices = np.arange(len(weights))
        ax10.plot(history_indices, weights, color='steelblue', linewidth=2, linestyle='-')
        ax10.scatter(history_indices, weights, color='steelblue', s=40, zorder=3)
        ax10.scatter([len(weights)-1], [weights[-1]], color='orange', s=100, marker='o', edgecolors='black', linewidth=1.5, zorder=4, label='Current Input')
        ax10.text(len(weights)-1, weights[-1], '  Now', color='black', ha='left', va='center', fontweight='bold')
        ax10.invert_xaxis()
        ax10.grid(True, alpha=0.3, axis='y')
        ax10.legend(loc='upper right')
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 11 ---
        print("Generating Figure 11: Statistical Distribution of Filtering Errors (6-DOF)")
        try:
            raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            overall_raw_vel = np.linalg.norm(raw_vel_matrix, axis=0)
            overall_smooth_vel = np.linalg.norm(smooth_vel_matrix, axis=0)
        except AttributeError:
            overall_raw_vel = np.zeros(len(steps))
            overall_smooth_vel = np.zeros(len(steps))
        residuals = overall_raw_vel - overall_smooth_vel
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        ax11.set_title("Statistical Distribution of Filtering Errors (6-DOF)", fontsize=14)
        ax11.set_xlabel("Overall Velocity Residual Error (6-DOF L2 Norm)")
        ax11.set_ylabel("Frequency Count")
        n, bins, patches = ax11.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        sigma = np.std(residuals)
        if sigma < 1e-9:
            mu = np.mean(residuals)
            ax11.plot([mu, mu], [0, np.max(n)], 'r--', linewidth=2, label='Constant Signal')
        else:
            mu, sigma = norm.fit(residuals)
            best_fit_line = norm.pdf(bins, mu, sigma)
            ax11.plot(bins, best_fit_line * len(residuals) * (bins[1]-bins[0]), 'r--', linewidth=2, label=rf'Normal Fit ($\mu={mu:.3f}, \sigma={sigma:.3f}$)')
        ax11.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax11.legend(loc='upper right')
        ax11.grid(True, alpha=0.3)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 12 ---
        print("Generating Figure 12: Frequency Attenuation (Left, Right & Average - All 6-DOF)")
        from matplotlib.gridspec import GridSpec
        fig12 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig12)
        ax12a = fig12.add_subplot(gs[0, 0]) 
        ax12b = fig12.add_subplot(gs[0, 1]) 
        ax12c = fig12.add_subplot(gs[1, :]) 
        fig12.suptitle("Frequency Attenuation (6-DOF L2 Norm)", fontsize=15)
        def plot_psd_subplot(ax, raw_vel, smooth_vel, title, label_prefix):
            freqs_raw, psd_raw = scipy.signal.welch(raw_vel, fs=60.0, nperseg=64)
            freqs_smooth, psd_smooth = scipy.signal.welch(smooth_vel, fs=60.0, nperseg=64)
            psd_raw_safe = np.maximum(psd_raw, 1e-20)
            psd_smooth_safe = np.maximum(psd_smooth, 1e-20)
            avg_psd_raw = np.mean(psd_raw_safe)
            avg_psd_smooth = np.mean(psd_smooth_safe)
            reduction_pct = ((avg_psd_raw - avg_psd_smooth) / avg_psd_raw * 100) if avg_psd_raw > 1e-9 else 0.0
            ax.semilogy(freqs_raw, psd_raw_safe, label=f'{label_prefix} Raw (Avg PSD: {avg_psd_raw:.7f})', color='red', alpha=0.6)
            ax.semilogy(freqs_smooth, psd_smooth_safe, label=f'{label_prefix} Smooth (Avg PSD: {avg_psd_smooth:.7f}, Red: {reduction_pct:.1f}%)', color='blue', linewidth=2)
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Power Spectral Density (dB)", fontsize=15)
            ax.set_xlabel("Frequency (Hz)", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlim([0, 30])
            ax.tick_params(axis='both', which='major', labelsize=15)
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            l_overall_raw_vel = np.linalg.norm(l_raw_vel_matrix, axis=0)
            l_overall_smooth_vel = np.linalg.norm(l_smooth_vel_matrix, axis=0)
            r_overall_raw_vel = np.linalg.norm(r_raw_vel_matrix, axis=0)
            r_overall_smooth_vel = np.linalg.norm(r_smooth_vel_matrix, axis=0)
            avg_overall_raw_vel = (l_overall_raw_vel + r_overall_raw_vel) / 2.0
            avg_overall_smooth_vel = (l_overall_smooth_vel + r_overall_smooth_vel) / 2.0
        except (AttributeError, IndexError):
            l_overall_raw_vel = np.zeros(len(steps))
            l_overall_smooth_vel = np.zeros(len(steps))
            r_overall_raw_vel = np.zeros(len(steps))
            r_overall_smooth_vel = np.zeros(len(steps))
            avg_overall_raw_vel = np.zeros(len(steps))
            avg_overall_smooth_vel = np.zeros(len(steps))
        plot_psd_subplot(ax12a, l_overall_raw_vel, l_overall_smooth_vel, "Left Arm (6-DOF)", "Left")
        plot_psd_subplot(ax12b, r_overall_raw_vel, r_overall_smooth_vel, "Right Arm (6-DOF)", "Right")
        plot_psd_subplot(ax12c, avg_overall_raw_vel, avg_overall_smooth_vel, "Average (Left + Right Arms)", "Avg (L+R)")
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # --- FIGURE 13 ---
        print("Generating Figure 13: Velocity Profile (Left, Right & Average - All 6-DOF)")
        fig13 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig13)
        ax13a = fig13.add_subplot(gs[0, 0]) 
        ax13b = fig13.add_subplot(gs[0, 1]) 
        ax13c = fig13.add_subplot(gs[1, :]) 
        fig13.suptitle("Velocity Profile Analysis (6-DOF L2 Norm)", fontsize=15)
        def plot_velocity_subplot(ax, raw_vel, smooth_vel, title, label_prefix):
            avg_raw = np.mean(raw_vel)
            avg_smooth = np.mean(smooth_vel)
            pct_change = ((avg_raw - avg_smooth) / avg_raw) * 100 if avg_raw > 1e-9 else 0.0
            ax.plot(steps[:len(raw_vel)], raw_vel, label=f'{label_prefix} Raw (Avg: {avg_raw:.4f})', color='red', alpha=0.3, linewidth=1)
            ax.plot(steps[:len(smooth_vel)], smooth_vel, label=f'{label_prefix} Smooth (Avg: {avg_smooth:.4f}, Chg: {pct_change:.1f}%)', color='green', linewidth=1.5)
            ax.fill_between(steps[:len(smooth_vel)], smooth_vel, 0, color='green', alpha=0.1)
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Overall Velocity Magnitude", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            l_vel_norm = np.linalg.norm(l_raw_vel_matrix, axis=0)
            l_smooth_vel_norm = np.linalg.norm(l_smooth_vel_matrix, axis=0)
            r_vel_norm = np.linalg.norm(r_raw_vel_matrix, axis=0)
            r_smooth_vel_norm = np.linalg.norm(r_smooth_vel_matrix, axis=0)
            avg_vel_norm = (l_vel_norm + r_vel_norm) / 2.0
            avg_smooth_vel_norm = (l_smooth_vel_norm + r_smooth_vel_norm) / 2.0
        except (AttributeError, IndexError):
            l_vel_norm = np.zeros(len(steps))
            l_smooth_vel_norm = np.zeros(len(steps))
            r_vel_norm = np.zeros(len(steps))
            r_smooth_vel_norm = np.zeros(len(steps))
            avg_vel_norm = np.zeros(len(steps))
            avg_smooth_vel_norm = np.zeros(len(steps))
        plot_velocity_subplot(ax13a, l_vel_norm, l_smooth_vel_norm, "Left Arm (6-DOF)", "Left")
        plot_velocity_subplot(ax13b, r_vel_norm, r_smooth_vel_norm, "Right Arm (6-DOF)", "Right")
        avg_r = np.mean(avg_vel_norm)
        avg_s = np.mean(avg_smooth_vel_norm)
        pct_change = ((avg_r - avg_s) / avg_r) * 100 if avg_r > 1e-9 else 0.0
        ax13c.plot(steps[:len(avg_vel_norm)], avg_vel_norm, label=f'Avg Raw (L+R) (Avg: {avg_r:.4f})', color='purple', alpha=0.3, linewidth=1)
        ax13c.plot(steps[:len(avg_smooth_vel_norm)], avg_smooth_vel_norm, label=f'Avg Smooth (L+R) (Avg: {avg_s:.4f}, Chg: {pct_change:.1f}%)', color='blue', linewidth=1.5)
        ax13c.fill_between(steps[:len(avg_smooth_vel_norm)], avg_smooth_vel_norm, 0, color='blue', alpha=0.1)
        ax13c.set_title("Average Velocity (Left + Right Arms)", fontsize=15)
        ax13c.set_xlabel("Time Steps", fontsize=15)
        ax13c.set_ylabel("Average Velocity Magnitude", fontsize=15)
        ax13c.legend(loc='upper right', fontsize=15)
        ax13c.tick_params(axis='both', which='major', labelsize=15)
        ax13c.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # --- FIGURE 14 ---
        print("Generating Figure 14: Input-Output Hysteresis (Left, Right & Avg - All 6-DOF)")
        from matplotlib.gridspec import GridSpec
        fig14 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig14)
        ax14a = fig14.add_subplot(gs[0, 0]) 
        ax14b = fig14.add_subplot(gs[0, 1]) 
        ax14c = fig14.add_subplot(gs[1, :]) 
        fig14.suptitle("Input-Output Hysteresis-changes of lags (6-DOF L2 Norm)", fontsize=15)
        def plot_hysteresis_subplot(ax, haptic_matrix, robot_matrix, title, label_prefix):
            haptic_norm = np.linalg.norm(haptic_matrix, axis=0)
            robot_norm = np.linalg.norm(robot_matrix, axis=0)
            mean_haptic_val = np.mean(haptic_norm)
            mean_robot_val = np.mean(robot_norm)
            limit = max(np.max(haptic_norm), np.max(robot_norm)) * 1.1
            limit = max(limit, 0.1)
            ax.plot([0, limit], [0, limit], 'k--', alpha=0.5, label=f'Ideal Tracking ({label_prefix} Raw: {mean_haptic_val:.4f}, Robot: {mean_robot_val:.4f})')
            sc = ax.scatter(haptic_norm, robot_norm, c=steps, cmap='plasma', s=10, alpha=0.6)
            ax.set_title(title, fontsize=15)
            ax.set_xlabel(f"{label_prefix} Haptic Pos (L2 Norm)", fontsize=15)
            ax.set_ylabel(f"{label_prefix} Robot Pos (L2 Norm)", fontsize=15)
            ax.set_xlim([0, limit])
            ax.set_ylim([0, limit])
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
            return sc
        try:
            l_haptic_matrix = np.array([list(self.viz.left_h_pos_hist[i]) for i in range(6)])
            l_robot_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_haptic_matrix = np.array([list(self.viz.right_h_pos_hist[i]) for i in range(6)])
            r_robot_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            avg_haptic_norm = (np.linalg.norm(l_haptic_matrix, axis=0) + np.linalg.norm(r_haptic_matrix, axis=0)) / 2.0
            avg_robot_norm = (np.linalg.norm(l_robot_matrix, axis=0) + np.linalg.norm(r_robot_matrix, axis=0)) / 2.0
        except (AttributeError, IndexError):
            l_haptic_matrix = np.zeros((6, len(steps)))
            l_robot_matrix = np.zeros((6, len(steps)))
            r_haptic_matrix = np.zeros((6, len(steps)))
            r_robot_matrix = np.zeros((6, len(steps)))
            avg_haptic_norm = np.zeros(len(steps))
            avg_robot_norm = np.zeros(len(steps))
        sc_l = plot_hysteresis_subplot(ax14a, l_haptic_matrix, l_robot_matrix, "Left Arm (6-DOF)", "Left")
        sc_r = plot_hysteresis_subplot(ax14b, r_haptic_matrix, r_robot_matrix, "Right Arm (6-DOF)", "Right")
        limit_avg = max(np.max(avg_haptic_norm), np.max(avg_robot_norm)) * 1.1
        limit_avg = max(limit_avg, 0.1)
        mean_haptic_avg = np.mean(avg_haptic_norm)
        mean_robot_avg = np.mean(avg_robot_norm)
        ax14c.plot([0, limit_avg], [0, limit_avg], 'k--', alpha=0.5, label=f'Ideal Tracking (Avg Raw: {mean_haptic_avg:.4f}, Avg Robot: {mean_robot_avg:.4f})')
        sc_avg = ax14c.scatter(avg_haptic_norm, avg_robot_norm, c=steps, cmap='plasma', s=10, alpha=0.6)
        ax14c.set_title("Average Position (Left + Right Arms)", fontsize=15)
        ax14c.set_xlabel("Avg Haptic Pos (L2 Norm)", fontsize=15)
        ax14c.set_ylabel("Avg Robot Pos (L2 Norm)", fontsize=15)
        ax14c.set_xlim([0, limit_avg])
        ax14c.set_ylim([0, limit_avg])
        ax14c.legend(loc='upper right', fontsize=15)
        ax14c.tick_params(axis='both', which='major', labelsize=15)
        ax14c.grid(True, alpha=0.3)
        fig14.colorbar(sc_avg, ax=ax14c, label='Time Step')
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 15 ---
        print("Generating Figure 15:Cumulative Control Effort (Left, Right & Avg - All 6-DOF)")
        fig15 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig15)
        ax15a = fig15.add_subplot(gs[0, 0]) 
        ax15b = fig15.add_subplot(gs[0, 1]) 
        ax15c = fig15.add_subplot(gs[1, :]) 
        fig15.suptitle("Cumulative Control Effort (6-DOF L2 Norm)", fontsize=15)
        def plot_effort_subplot(ax, raw_vel_matrix, smooth_vel_matrix, title, label_prefix):
            raw_vel = np.linalg.norm(raw_vel_matrix, axis=0)
            smooth_vel = np.linalg.norm(smooth_vel_matrix, axis=0)
            effort_raw = np.cumsum(raw_vel)
            effort_smooth = np.cumsum(smooth_vel)
            avg_raw = np.mean(raw_vel)
            avg_smooth = np.mean(smooth_vel)
            ax.plot(steps[:len(raw_vel)], effort_raw, label=f'{label_prefix} Raw (Avg Vel: {avg_raw:.4f})', color='red', alpha=0.6, linestyle='--')
            ax.plot(steps[:len(smooth_vel)], effort_smooth, label=f'{label_prefix} Smooth (Avg Vel: {avg_smooth:.4f})', color='blue', linewidth=2)
            final_diff = effort_raw[-1] - effort_smooth[-1]
            ax.fill_between(steps[:len(effort_smooth)], effort_smooth, effort_raw, color='red', alpha=0.1, label=f'Energy Saved: {final_diff:.2f}')
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Accumulated Effort (Arbitrary Units)", fontsize=15)
            ax.legend(loc='upper left', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
            return effort_raw, effort_smooth
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            l_raw_vel_norm = np.linalg.norm(l_raw_vel_matrix, axis=0)
            l_smooth_vel_norm = np.linalg.norm(l_smooth_vel_matrix, axis=0)
            r_raw_vel_norm = np.linalg.norm(r_raw_vel_matrix, axis=0)
            r_smooth_vel_norm = np.linalg.norm(r_smooth_vel_matrix, axis=0)
            avg_vel_norm = (l_raw_vel_norm + r_raw_vel_norm) / 2.0
            avg_smooth_vel_norm = (l_smooth_vel_norm + r_smooth_vel_norm) / 2.0
        except (AttributeError, IndexError):
            l_raw_vel_norm = np.zeros(len(steps))
            l_smooth_vel_norm = np.zeros(len(steps))
            r_raw_vel_norm = np.zeros(len(steps))
            r_smooth_vel_norm = np.zeros(len(steps))
            avg_vel_norm = np.zeros(len(steps))
            avg_smooth_vel_norm = np.zeros(len(steps))
        plot_effort_subplot(ax15a, l_raw_vel_matrix, l_smooth_vel_matrix, "Left Arm (6-DOF)", "Left")
        plot_effort_subplot(ax15b, r_raw_vel_matrix, r_smooth_vel_matrix, "Right Arm (6-DOF)", "Right")
        effort_avg_raw = np.cumsum(avg_vel_norm)
        effort_avg_smooth = np.cumsum(avg_smooth_vel_norm)
        avg_r = np.mean(avg_vel_norm)
        avg_s = np.mean(avg_smooth_vel_norm)
        final_diff = effort_avg_raw[-1] - effort_avg_smooth[-1]
        ax15c.plot(steps[:len(avg_vel_norm)], effort_avg_raw, label=f'Avg Raw (L+R) (Avg Vel: {avg_r:.4f})', color='purple', alpha=0.6, linestyle='--')
        ax15c.plot(steps[:len(avg_smooth_vel_norm)], effort_avg_smooth, label=f'Avg Smooth (L+R) (Avg Vel: {avg_s:.4f})', color='green', linewidth=2)
        ax15c.fill_between(steps[:len(avg_smooth_vel_norm)], effort_avg_smooth, effort_avg_raw, color='red', alpha=0.1, label=f'Energy Saved: {final_diff:.2f}')
        ax15c.set_title("Average Cumulative Effort (Left + Right Arms)", fontsize=15)
        ax15c.set_ylabel("Accumulated Effort (Arbitrary Units)", fontsize=15)
        ax15c.set_xlabel("Time Steps", fontsize=15)
        ax15c.legend(loc='upper left', fontsize=15)
        ax15c.tick_params(axis='both', which='major', labelsize=15)
        ax15c.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 16 ---
        print("Generating Figure 16: Motion Coupling (6-DOF Correlation Matrix)")
        fig16, (ax16a, ax16b, ax16c) = plt.subplots(1, 3, figsize=(18, 6))
        fig16.suptitle("Motion Coupling (6-DOF Correlation Matrix)", fontsize=15)
        def plot_corr_subplot(ax, pos_matrix, title):
            data_matrix = pos_matrix + np.random.normal(0, 1e-9, size=pos_matrix.shape)
            corr_matrix = np.corrcoef(data_matrix)
            cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            for (i, j), val in np.ndenumerate(corr_matrix):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontweight='bold')
            ax.set_title(title, fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
        try:
            l_pos_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_pos_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            avg_pos_matrix = (l_pos_matrix + r_pos_matrix) / 2.0
        except (AttributeError, IndexError):
            l_pos_matrix = np.zeros((6, len(steps)))
            r_pos_matrix = np.zeros((6, len(steps)))
            avg_pos_matrix = np.zeros((6, len(steps)))
        plot_corr_subplot(ax16a, l_pos_matrix, "Left Arm (6-DOF)")
        plot_corr_subplot(ax16b, r_pos_matrix, "Right Arm (6-DOF)")
        plot_corr_subplot(ax16c, avg_pos_matrix, "Average (L+R)")
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # --- FIGURE 17 ---
        print("Generating Figure 17: Scalogram (Time-Frequency Intensity - 6-DOF)")
        fig17, (ax17a, ax17b, ax17c) = plt.subplots(1, 3, figsize=(18, 6))
        fig17.suptitle("Figure 17: Scalogram (6-DOF L2 Norm Analysis)", fontsize=15)
        def ricker_wavelet(points, a):
            A = 2 / (np.sqrt(3 * a) * np.pi ** 0.25)
            wsq = a**2
            vec = np.arange(0, points) - (points - 1.0) / 2
            xsq = vec**2
            mod = (1 - xsq / wsq) * np.exp(-xsq / (2 * wsq))
            return A * mod
        def plot_scalogram_subplot(ax, data, title):
            subset_slice = slice(0, min(200, len(data)))
            sig = data[subset_slice]
            widths = np.arange(1, 31)
            cwtmatr = []
            for w in widths:
                wavelet = ricker_wavelet(w, w)
                cwt_row = scipy.signal.convolve(sig, wavelet, mode='same')
                cwtmatr.append(cwt_row)
            cwtmatr = np.array(cwtmatr)
            mesh = ax.pcolormesh(np.arange(sig.shape[0]), widths, np.abs(cwtmatr), cmap='viridis', shading='auto')
            fig17.colorbar(mesh, ax=ax, label='Magnitude')
            ax.set_title(title, fontsize=15)
            ax.set_xlabel("Time Steps", fontsize=15)
            ax.set_ylabel("Wavelet Scale (Width)", fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
        try:
            l_pos_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_pos_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            l_overall_pos = np.linalg.norm(l_pos_matrix, axis=0)
            r_overall_pos = np.linalg.norm(r_pos_matrix, axis=0)
            avg_overall_pos = (l_overall_pos + r_overall_pos) / 2.0
        except (AttributeError, IndexError):
            l_overall_pos = np.zeros(len(steps))
            r_overall_pos = np.zeros(len(steps))
            avg_overall_pos = np.zeros(len(steps))
        plot_scalogram_subplot(ax17a, l_overall_pos, "Left Arm (6-DOF)")
        plot_scalogram_subplot(ax17b, r_overall_pos, "Right Arm (6-DOF)")
        plot_scalogram_subplot(ax17c, avg_overall_pos, "Average (Left + Right)")
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 18 ---
        print("Generating Figure 18: Autocorrelation Function (Signal Memory - 6-DOF)")
        def autocorr(x):
            result = np.correlate(x, x, mode='full')
            return result[len(result)//2:]
        def plot_acf_subplot(ax, raw_vel, smooth_vel, title, label_prefix):
            if len(raw_vel) < 2:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(title)
                return
            raw_norm = (raw_vel - np.mean(raw_vel)) / (np.std(raw_vel) + 1e-8)
            smooth_norm = (smooth_vel - np.mean(smooth_vel)) / (np.std(smooth_vel) + 1e-8)
            raw_acf = autocorr(raw_norm)
            smooth_acf = autocorr(smooth_norm)
            lags = np.arange(len(raw_acf) // 2)
            raw_acf_sliced = raw_acf[:len(lags)]
            smooth_acf_sliced = smooth_acf[:len(lags)]
            avg_raw_acf = np.mean(raw_acf_sliced)
            avg_smooth_acf = np.mean(smooth_acf_sliced)
            ax.plot(lags, raw_acf_sliced, label=f'{label_prefix} Raw [Short Memory] (Avg ACF: {avg_raw_acf:.3f})', color='red', alpha=0.6)
            ax.plot(lags, smooth_acf_sliced, label=f'{label_prefix} SmoothOps [Long Memory] (Avg ACF: {avg_smooth_acf:.3f})', color='blue', linewidth=2)
            ax.fill_between(lags, raw_acf_sliced, smooth_acf_sliced, color='gray', alpha=0.1)
            ax.set_title(title, fontsize=15)
            ax.set_xlabel("Time Lags", fontsize=15)
            ax.set_ylabel("Correlation Coefficient", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 50])
        fig18, (ax18a, ax18b, ax18c) = plt.subplots(1, 3, figsize=(18, 6))
        fig18.suptitle("Figure 18: Autocorrelation Function (6-DOF L2 Norm)", fontsize=15)
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            l_overall_raw = np.linalg.norm(l_raw_vel_matrix, axis=0)
            l_overall_smooth = np.linalg.norm(l_smooth_vel_matrix, axis=0)
            r_overall_raw = np.linalg.norm(r_raw_vel_matrix, axis=0)
            r_overall_smooth = np.linalg.norm(r_smooth_vel_matrix, axis=0)
            avg_overall_raw = (l_overall_raw + r_overall_raw) / 2.0
            avg_overall_smooth = (l_overall_smooth + r_overall_smooth) / 2.0
        except (AttributeError, IndexError):
            l_overall_raw = np.zeros(len(steps))
            l_overall_smooth = np.zeros(len(steps))
            r_overall_raw = np.zeros(len(steps))
            r_overall_smooth = np.zeros(len(steps))
            avg_overall_raw = np.zeros(len(steps))
            avg_overall_smooth = np.zeros(len(steps))
        plot_acf_subplot(ax18a, l_overall_raw, l_overall_smooth, "Left Arm (6-DOF)", "Left")
        plot_acf_subplot(ax18b, r_overall_raw, r_overall_smooth, "Right Arm (6-DOF)", "Right")
        plot_acf_subplot(ax18c, avg_overall_raw, avg_overall_smooth, "Average (Left + Right Arms)", "Avg (L+R)")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
                
        # --- FIGURE 19 ---
        print("Generating Figure 19: System Performance Quantification (6-DOF L2 Norm)")
        def calculate_performance_metrics(raw_vel, smooth_vel):
            rmse = np.sqrt(np.mean((raw_vel - smooth_vel)**2))
            mae = np.mean(np.abs(raw_vel - smooth_vel))
            p2p_raw = np.max(raw_vel) - np.min(raw_vel)
            p2p_smooth = np.max(smooth_vel) - np.min(smooth_vel)
            p2p_reduction = (1 - p2p_smooth / p2p_raw) * 100 if p2p_raw > 1e-9 else 0.0
            var_smooth = np.var(smooth_vel)
            var_noise = np.var(raw_vel - smooth_vel)
            snr_db = 90.0
            if var_noise > 1e-9:
                 snr_db = 10 * np.log10(var_smooth / var_noise)
            return rmse, mae, snr_db, p2p_reduction
        def plot_performance_subplot(ax, raw_vel, smooth_vel, title):
            rmse, mae, snr_db, p2p_red = calculate_performance_metrics(raw_vel, smooth_vel)
            labels = ['RMSE', 'MAE', 'SNR (dB)', 'P2P Red (%)']
            values = [rmse, mae, snr_db, p2p_red]
            colors = ['coral', 'orange', 'skyblue', 'lightgreen']
            bars = ax.bar(labels, values, color=colors, edgecolor='black')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=15)
            ax.set_title(title, fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, max(100, np.max(values)*1.1)])
        fig19, (ax19a, ax19b, ax19c) = plt.subplots(1, 3, figsize=(18, 6))
        fig19.suptitle("System Performance Quantification (6-DOF L2 Norm)", fontsize=15)
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            l_raw_overall = np.linalg.norm(l_raw_vel_matrix, axis=0)
            l_smooth_overall = np.linalg.norm(l_smooth_vel_matrix, axis=0)
            r_raw_overall = np.linalg.norm(r_raw_vel_matrix, axis=0)
            r_smooth_overall = np.linalg.norm(r_smooth_vel_matrix, axis=0)
            avg_raw_overall = (l_raw_overall + r_raw_overall) / 2.0
            avg_smooth_overall = (l_smooth_overall + r_smooth_overall) / 2.0
        except (AttributeError, IndexError):
            l_raw_overall = np.zeros(len(steps))
            l_smooth_overall = np.zeros(len(steps))
            r_raw_overall = np.zeros(len(steps))
            r_smooth_overall = np.zeros(len(steps))
            avg_raw_overall = np.zeros(len(steps))
            avg_smooth_overall = np.zeros(len(steps))
        plot_performance_subplot(ax19a, l_raw_overall, l_smooth_overall, "Left Arm (6-DOF)")
        plot_performance_subplot(ax19b, r_raw_overall, r_smooth_overall, "Right Arm (6-DOF)")
        plot_performance_subplot(ax19c, avg_raw_overall, avg_smooth_overall, "Average (Left + Right)")
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 20 ---
        print("Generating Figure 20: Magnitude Coherence (Frequency Tracking Fidelity)")
        f_coh, Cxy = scipy.signal.coherence(raw_vel_3, smooth_vel_3, fs=60.0, nperseg=64)
        fig20, ax20 = plt.subplots(figsize=(10, 6))
        ax20.set_title("Magnitude Coherence (Frequency Tracking Fidelity)", fontsize=14)
        ax20.set_xlabel("Frequency (Hz)")
        ax20.set_ylabel("Coherence Magnitude")
        ax20.plot(f_coh, Cxy, color='purple', linewidth=2)
        ax20.fill_between(f_coh, Cxy, 0, color='purple', alpha=0.2)
        ax20.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='High Tracking (>0.9)')
        ax20.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Noise Rejection (<0.1)')
        ax20.set_xlim([0, 30])
        ax20.set_ylim([0, 1.1])
        ax20.legend(loc='upper right')
        ax20.grid(True, alpha=0.3)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 21 ---
        print("Generating Figure 21: Error Dynamics (Stability Analysis - 6-DOF L2 Norm)")
        fig21, (ax21a, ax21b, ax21c) = plt.subplots(1, 3, figsize=(18, 6))
        fig21.suptitle("Error Dynamics (6-DOF L2 Norm)", fontsize=16)
        def plot_dynamics_subplot(ax, raw_vel, smooth_vel, title, label_prefix):
            if len(raw_vel) < 2:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(title)
                return
            residuals = raw_vel - smooth_vel
            d_err = np.gradient(residuals)
            sc = ax.scatter(residuals, d_err, c=steps, cmap='plasma', s=10, alpha=0.6)
            ax.scatter([0], [0], color='red', s=100, marker='x', label='Target Stability')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_title(title)
            ax.set_xlabel(f"{label_prefix} Residual Magnitude (6-DOF)")
            ax.set_ylabel(f"{label_prefix} Residual Rate of Change")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            return sc
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            l_overall_raw = np.linalg.norm(l_raw_vel_matrix, axis=0)
            l_overall_smooth = np.linalg.norm(l_smooth_vel_matrix, axis=0)
            r_overall_raw = np.linalg.norm(r_raw_vel_matrix, axis=0)
            r_overall_smooth = np.linalg.norm(r_smooth_vel_matrix, axis=0)
            avg_overall_raw = (l_overall_raw + r_overall_raw) / 2.0
            avg_overall_smooth = (l_overall_smooth + r_overall_smooth) / 2.0
        except (AttributeError, IndexError):
            l_overall_raw = np.zeros(len(steps))
            l_overall_smooth = np.zeros(len(steps))
            r_overall_raw = np.zeros(len(steps))
            r_overall_smooth = np.zeros(len(steps))
            avg_overall_raw = np.zeros(len(steps))
            avg_overall_smooth = np.zeros(len(steps))
        sc_l = plot_dynamics_subplot(ax21a, l_overall_raw, l_overall_smooth, "Left Arm (6-DOF)", "Left")
        sc_r = plot_dynamics_subplot(ax21b, r_overall_raw, r_overall_smooth, "Right Arm (6-DOF)", "Right")
        sc_avg = plot_dynamics_subplot(ax21c, avg_overall_raw, avg_overall_smooth, "Average (Left + Right)", "Avg (L+R)")
        fig21.colorbar(sc_avg, ax=ax21c, label='Time Step')
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 22 ---
        print("Generating Figure 22: Shannon Entropy (6-DOF Information Content)")
        from matplotlib.gridspec import GridSpec
        fig22, (ax22a, ax22b, ax22c) = plt.subplots(1, 3, figsize=(18, 6))
        fig22.suptitle("Shannon Entropy (6-DOF L2 Norm)", fontsize=15)
        def calculate_entropy(signal, bin_count=20):
            sig_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-9)
            hist, _ = np.histogram(sig_norm, bins=bin_count, density=True)
            p = hist / np.sum(hist) + 1e-10
            return -np.sum(p * np.log2(p))
        def plot_entropy_subplot(ax, haptic_matrix, robot_matrix, title, label_prefix):
            haptic_vel_norm = np.linalg.norm(haptic_matrix, axis=0)
            robot_vel_norm = np.linalg.norm(robot_matrix, axis=0)
            window = 30
            entropy_raw = []; entropy_smooth = []; time_indices = []
            data_len = len(haptic_vel_norm)
            if data_len > window:
                for i in range(window, data_len - window):
                    e_r = calculate_entropy(haptic_vel_norm[i-window:i+window])
                    e_s = calculate_entropy(robot_vel_norm[i-window:i+window])
                    entropy_raw.append(e_r)
                    entropy_smooth.append(e_s)
                    time_indices.append(steps[i])
            avg_ent_r = np.mean(entropy_raw)
            avg_ent_s = np.mean(entropy_smooth)
            ax.plot(time_indices, entropy_raw, label=f'{label_prefix} Raw (Avg: {avg_ent_r:.3f})', color='red', alpha=0.6, linewidth=1)
            ax.plot(time_indices, entropy_smooth, label=f'{label_prefix} Smooth (Avg: {avg_ent_s:.3f})', color='blue', linewidth=2)
            ax.fill_between(time_indices, entropy_smooth, entropy_raw, color='green', alpha=0.1, label='Redundant Noise Removed')
            ax.set_title(title, fontsize=15)
            ax.set_xlabel("Time Steps", fontsize=15)
            ax.set_ylabel("Entropy (Bits)", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
            return entropy_raw, entropy_smooth
        try:
            l_h_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_s_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_h_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_s_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            avg_h_vel_norm = (np.linalg.norm(l_h_vel_matrix, axis=0) + np.linalg.norm(r_h_vel_matrix, axis=0)) / 2.0
            avg_s_vel_norm = (np.linalg.norm(l_s_vel_matrix, axis=0) + np.linalg.norm(r_s_vel_matrix, axis=0)) / 2.0
        except (AttributeError, IndexError):
            l_h_vel_matrix = np.zeros((6, len(steps)))
            l_s_vel_matrix = np.zeros((6, len(steps)))
            r_h_vel_matrix = np.zeros((6, len(steps)))
            r_s_vel_matrix = np.zeros((6, len(steps)))
            avg_h_vel_norm = np.zeros(len(steps))
            avg_s_vel_norm = np.zeros(len(steps))
        plot_entropy_subplot(ax22a, l_h_vel_matrix, l_s_vel_matrix, "Left Arm (6-DOF)", "Left")
        plot_entropy_subplot(ax22b, r_h_vel_matrix, r_s_vel_matrix, "Right Arm (6-DOF)", "Right")
        window = 30
        sys_entropy_raw = []; sys_entropy_smooth = []; sys_time_indices = []
        if len(avg_h_vel_norm) > window:
            for i in range(window, len(avg_h_vel_norm) - window):
                e_r = calculate_entropy(avg_h_vel_norm[i-window:i+window])
                e_s = calculate_entropy(avg_s_vel_norm[i-window:i+window])
                sys_entropy_raw.append(e_r); sys_entropy_smooth.append(e_s); sys_time_indices.append(steps[i])
        avg_sys_r = np.mean(sys_entropy_raw)
        avg_sys_s = np.mean(sys_entropy_smooth)
        ax22c.plot(sys_time_indices, sys_entropy_raw, label=f'Avg Raw (L+R) (Avg: {avg_sys_r:.3f})', color='purple', alpha=0.6, linewidth=1)
        ax22c.plot(sys_time_indices, sys_entropy_smooth, label=f'Avg Smooth (L+R) (Avg: {avg_sys_s:.3f})', color='blue', linewidth=2)
        ax22c.fill_between(sys_time_indices, sys_entropy_smooth, sys_entropy_raw, color='green', alpha=0.1, label='Redundant Noise Removed')
        ax22c.set_title("Average Entropy (Left + Right Arms)", fontsize=15)
        ax22c.set_xlabel("Time Steps", fontsize=15)
        ax22c.set_ylabel("Entropy (Bits)", fontsize=15)
        ax22c.legend(loc='upper right', fontsize=15)
        ax22c.tick_params(axis='both', which='major', labelsize=15)
        ax22c.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 23 ---
        print("Generating Figure 23: 6-DOF Action Analysis (Smoothed Intent vs Raw Haptic)")
        fig23, (ax23a, ax23b, ax23c) = plt.subplots(1, 3, figsize=(18, 6))
        fig23.suptitle("Figure 23: 6-DOF Action Analysis (Smoothed Intent vs Raw Haptic)", fontsize=15)
        def compute_6dof_norm(matrix):
            return np.linalg.norm(matrix, axis=0)
        def plot_intent_6dof_subplot(ax, raw_matrix, intent_matrix, title, label_prefix):
            raw_norm = compute_6dof_norm(raw_matrix)
            intent_norm = compute_6dof_norm(intent_matrix)
            avg_raw = np.mean(raw_norm)
            avg_intent = np.mean(intent_norm)
            steps_arg = steps 
            ax.plot(steps_arg, raw_norm, label=f"Raw Haptic (Avg: {avg_raw:.3f})", color='gray', linestyle='--', alpha=0.6)
            ax.plot(steps_arg, intent_norm, label=f"Robot Transformer Model Intent (Avg: {avg_intent:.3f})", linewidth=1.5)
            ax.fill_between(steps_arg, intent_norm, raw_norm, color='skyblue', alpha=0.2, label='Planning Delta')
            ax.set_title(title, fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("6-DOF Magnitude (L2 Norm)", fontsize=15)
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
        except (AttributeError, IndexError):
            l_raw_vel_matrix = np.zeros((6, len(steps)))
            l_smooth_vel_matrix = np.zeros((6, len(steps)))
            r_raw_vel_matrix = np.zeros((6, len(steps)))
            r_smooth_vel_matrix = np.zeros((6, len(steps)))
        plot_intent_6dof_subplot(ax23a, l_raw_vel_matrix, l_smooth_vel_matrix, "Left Arm (6-DOF)", "Left")
        plot_intent_6dof_subplot(ax23b, r_raw_vel_matrix, r_smooth_vel_matrix, "Right Arm (6-DOF)", "Right")
        avg_raw_matrix = (l_raw_vel_matrix + r_raw_vel_matrix) / 2.0
        avg_smooth_matrix = (l_smooth_vel_matrix + r_smooth_vel_matrix) / 2.0
        plot_intent_6dof_subplot(ax23c, avg_raw_matrix, avg_smooth_matrix, "Average System (Left+Right)", "Avg")
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 24 ---
        print("Generating Figure 24: Explainability AI - Adaptive Compliance (6-DOF Speed)")
        from matplotlib.gridspec import GridSpec
        fig24 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig24)
        ax24a = fig24.add_subplot(gs[0, 0]) 
        ax24b = fig24.add_subplot(gs[0, 1]) 
        ax24c = fig24.add_subplot(gs[1, :]) 
        fig24.suptitle("Explainability - Adaptive Compliance (6-DOF Speed)", fontsize=15)
        def plot_compliance_subplot(ax, data, title, label_prefix):
            avg_comp = np.mean(data)
            ax.plot(steps[:len(data)], data, label=f'{label_prefix} Compliance (Avg: {avg_comp:.3f})', color='purple', linewidth=2)
            ax.fill_between(steps[:len(data)], data, 0, color='purple', alpha=0.2, label=f'Interaction Area')
            ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Neutral Threshold')
            ax.axhline(y=0.1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Soft Limit (Safety)')
            ax.axhline(y=0.9, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Stiff Limit (High Speed)')
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Compliance Factor (0=Stiff, 1=Soft)", fontsize=15)
            ax.set_xlabel("Time Steps", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])
        try:
            comp_data = np.array(list(self.viz.compliance_factor_hist))
        except (AttributeError, IndexError):
            comp_data = np.zeros(len(steps))
        plot_compliance_subplot(ax24a, comp_data, "Left Arm (System Stiffness)", "Left")
        plot_compliance_subplot(ax24b, comp_data, "Right Arm (System Stiffness)", "Right")
        plot_compliance_subplot(ax24c, comp_data, "Average System Stiffness (Left + Right)", "Avg")
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # --- FIGURE 25 ---
        print("Generating Figure 25: Explainability of - Kalman Filter Effect (6-DOF + % Change)")
        fig25, (ax25a, ax25b, ax25c) = plt.subplots(1, 3, figsize=(18, 6))
        fig25.suptitle("Kalman Filter Effect (6-DOF L2 Norm)", fontsize=15)
        def plot_kalman_subplot(ax, raw_matrix, corrected_matrix, title, label_prefix):
            raw_norm = np.linalg.norm(raw_matrix, axis=0)
            corrected_norm = np.linalg.norm(corrected_matrix, axis=0)
            correction_delta = np.abs(corrected_norm - raw_norm)
            avg_raw_mag = np.mean(raw_norm)
            epsilon = 1e-9
            if avg_raw_mag < epsilon:
                avg_pct_change = 0.0
            else:
                pct_change = (correction_delta / avg_raw_mag) * 100
                avg_pct_change = np.mean(pct_change)
            avg_raw = np.mean(raw_norm)
            avg_corr = np.mean(corrected_norm)
            ax.plot(steps[:len(raw_norm)], raw_norm, label=f'{label_prefix} Raw Input (Avg: {avg_raw:.4f})', color='orange', alpha=0.5, linestyle='--')
            ax.plot(steps[:len(corrected_norm)], corrected_norm, label=f'{label_prefix} Kalman Output (Avg: {avg_corr:.4f})', color='blue', linewidth=1.5)
            ax.fill_between(steps[:len(corrected_norm)], corrected_norm, raw_norm, color='gray', alpha=0.1, label=f'Correction Delta (Avg Red: {avg_pct_change:.1f}%)')
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("6-DOF Velocity Magnitude", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_corr_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_corr_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            avg_raw_matrix = (l_raw_vel_matrix + r_raw_vel_matrix) / 2.0
            avg_corr_matrix = (l_corr_vel_matrix + r_corr_vel_matrix) / 2.0
        except (AttributeError, IndexError):
            l_raw_vel_matrix = np.zeros((6, len(steps)))
            l_corr_vel_matrix = np.zeros((6, len(steps)))
            r_raw_vel_matrix = np.zeros((6, len(steps)))
            r_corr_vel_matrix = np.zeros((6, len(steps)))
            avg_raw_matrix = np.zeros((6, len(steps)))
            avg_corr_matrix = np.zeros((6, len(steps)))
        plot_kalman_subplot(ax25a, l_raw_vel_matrix, l_corr_vel_matrix, "Left Arm (6-DOF)", "Left")
        plot_kalman_subplot(ax25b, r_raw_vel_matrix, r_corr_vel_matrix, "Right Arm (6-DOF)", "Right")
        plot_kalman_subplot(ax25c, avg_raw_matrix, avg_corr_matrix, "Average System (Left+Right)", "Avg (L+R)")
        ax25c.set_xlabel("Time Steps", fontsize=15)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 26 (RESTORED: Safety Limiter) ---
        print("Generating Figure 26: Explainability - Safety Limiter (6-DOF L2 Norm)")
        fig26, (ax26a, ax26b, ax26c) = plt.subplots(1, 3, figsize=(18, 6))
        fig26.suptitle("Safety Limiter Analysis (6-DOF L2 Norm)", fontsize=15)
        def plot_safety_subplot(ax, safe_matrix, title, label_prefix):
            safe_norm = np.linalg.norm(safe_matrix, axis=0)
            max_limit_per_joint = 0.005
            max_l2_limit = np.sqrt(6) * max_limit_per_joint
            avg_safe = np.mean(safe_norm)
            avg_limit_util = (avg_safe / max_l2_limit) * 100
            ax.plot(steps[:len(safe_norm)], safe_norm, label=f'{label_prefix} Clamp Mag (Avg: {avg_safe:.4f})', color='red', alpha=0.6)
            ax.axhline(y=max_l2_limit, color='black', linestyle='--', linewidth=2, label='Max System Limit')
            ax.fill_between(steps[:len(safe_norm)], safe_norm, 0, color='orange', alpha=0.1, label=f'Limit Utilization (Avg: {avg_limit_util:.1f}%)')
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Safety Clamp Magnitude (L2 Norm)", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
        try:
            # Pulling from the RESTORED safe_hist
            l_safe_matrix = np.array([list(self.viz.left_safe_hist[i]) for i in range(6)])
            r_safe_matrix = np.array([list(self.viz.right_safe_hist[i]) for i in range(6)])
            avg_safe_matrix = (l_safe_matrix + r_safe_matrix) / 2.0
        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 26. {e}")
            l_safe_matrix = np.zeros((6, len(steps)))
            r_safe_matrix = np.zeros((6, len(steps)))
            avg_safe_matrix = np.zeros((6, len(steps)))
        plot_safety_subplot(ax26a, l_safe_matrix, "Left Arm (6-DOF)", "Left")
        plot_safety_subplot(ax26b, r_safe_matrix, "Right Arm (6-DOF)", "Right")
        plot_safety_subplot(ax26c, avg_safe_matrix, "Average System (Left+Right)", "Avg (L+R)")
        ax26c.set_xlabel("Time Steps", fontsize=15)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 27 (NEW: Fuzzy Force Estimation) ---
        print("Generating Figure 27: Explainability - Fuzzy Force Estimation (6-DOF L2 Norm)")
        fig27, (ax27a, ax27b, ax27c) = plt.subplots(1, 3, figsize=(18, 6))
        fig27.suptitle("Fuzzy Force Estimation (6-DOF L2 Norm)", fontsize=15)
        def plot_fuzzy_subplot(ax, fuzzy_matrix, title, label_prefix):
            fuzzy_norm = np.linalg.norm(fuzzy_matrix, axis=0)
            avg_fuzzy = np.mean(fuzzy_norm)
            ax.plot(steps[:len(fuzzy_norm)], fuzzy_norm, label=f'{label_prefix} Fuzzy Force (Avg: {avg_fuzzy:.4f})', color='red', alpha=0.6)
            ax.fill_between(steps[:len(fuzzy_norm)], fuzzy_norm, 0, color='orange', alpha=0.1, label=f'Compensation Magnitude')
            ax.set_title(title, fontsize=15)
            ax.set_ylabel("Force Estimation Magnitude", fontsize=15)
            ax.legend(loc='upper right', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid(True, alpha=0.3)
        try:
            # Pulling from the NEW fuzzy_hist
            l_fuzzy_matrix = np.array([list(self.viz.left_fuzzy_hist[i]) for i in range(6)])
            r_fuzzy_matrix = np.array([list(self.viz.right_fuzzy_hist[i]) for i in range(6)])
            avg_fuzzy_matrix = (l_fuzzy_matrix + r_fuzzy_matrix) / 2.0
        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 27. {e}")
            l_fuzzy_matrix = np.zeros((6, len(steps)))
            r_fuzzy_matrix = np.zeros((6, len(steps)))
            avg_fuzzy_matrix = np.zeros((6, len(steps)))
        plot_fuzzy_subplot(ax27a, l_fuzzy_matrix, "Left Arm (6-DOF)", "Left")
        plot_fuzzy_subplot(ax27b, r_fuzzy_matrix, "Right Arm (6-DOF)", "Right")
        plot_fuzzy_subplot(ax27c, avg_fuzzy_matrix, "Average System (Left+Right)", "Avg (L+R)")
        ax27c.set_xlabel("Time Steps", fontsize=15)
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        pp.close()

# =============================================================================
# 2. HAPTIC DEVICE SETUP
# =============================================================================

os.environ['GTDD_HOME'] = os.path.expanduser('~/.3dsystems')

def find_haptics_library():
    likely_path = '/home/shafiq/haptics_ws/drivers/openhaptics_3.4-0-developer-edition-amd64/usr/lib'
    if os.path.exists(os.path.join(likely_path, 'libHD.so')):
        return likely_path
    system_libs = ['/usr/lib', '/usr/local/lib', '/usr/lib/x86_64-linux-gnu']
    for path in system_libs:
        if os.path.exists(os.path.join(path, 'libHD.so')):
            return path
    search_root = '/home/shafiq/haptics_ws'
    for root, dirs, files in os.walk(search_root):
        if 'libHD.so' in files:
            return root
    return None

lib_path = find_haptics_library()

if lib_path:
    print(f"Found library at: {lib_path}")
    os.environ['LD_LIBRARY_PATH'] = lib_path
else:
    print("ERROR: Could not find libHD.so.")
    sys.exit(1)

try:
    ctypes.CDLL(os.path.join(lib_path, 'libHD.so'))
    print("Library loaded successfully.")
except Exception as e:
    print(f"Error loading library: {e}")
    sys.exit(1)

@dataclass
class DeviceState:
    position: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    gimbals: list = field(default_factory=list)
    full_joints: list = field(default_factory=list)
    btn_top: bool = False
    btn_bottom: bool = False
    force: list = field(default_factory=list)

device_state = DeviceState()

@hd_callback
def state_callback():
    global device_state
    try:
        motors = hd.get_joints()
        device_state.joints = [motors[0], motors[1], motors[2]]
        gimbals = hd.get_gimbals()
        device_state.gimbals = [gimbals[0], gimbals[1], gimbals[2]]
        device_state.full_joints = device_state.joints + device_state.gimbals
        btn_mask = hd.get_buttons()
        device_state.btn_top = (btn_mask & 1) != 0
        device_state.btn_bottom = (btn_mask & 2) != 0
        device_state.force = [0, 0, 0]
        hd.set_force(device_state.force)
    except Exception as e:
        pass

# =============================================================================
# 3. MUJOCO SIMULATION SETUP
# =============================================================================

XML_PATH = "/home/shafiq/Desktop/0ALOHA-ALL/mobile_aloha_sim-master/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"

def clamp(v, mn, mx): return np.clip(v, mn, mx)

try: 
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    print(f"Model loaded from {XML_PATH}")
except Exception as e:
    print(f"XML Error: {e}")
    sys.exit(1)

data = mujoco.MjData(model)

# --- RENDERER SETUP ---
print("Initializing Offscreen Renderer...")
renderer = None
cam_left_id = -1
cam_right_id = -1

try:
    renderer = mujoco.Renderer(model)
    if model.camera("fl_dabai"):
        cam_left_id = model.camera("fl_dabai").id
    if model.camera("fr_dabai"):
        cam_right_id = model.camera("fr_dabai").id
    print("Vision Display Ready")
except:
    print("Wrist cameras not found in XML.")

# Control Ranges
ctrl_ranges = np.array([
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
])

# Indices
right_arm_joint_names = [f"fr_joint{i}" for i in range(1, 9)]
right_arm_actuator_ctrl_indices = [model.actuator(name).id for name in right_arm_joint_names]
right_gripper_indices = [model.actuator("fr_joint7").id, model.actuator("fr_joint8").id]
right_body_id = model.body("fr_link6").id

left_arm_joint_names = [f"fl_joint{i}" for i in range(1, 9)]
left_arm_actuator_ctrl_indices = [model.actuator(name).id for name in left_arm_joint_names]
left_gripper_indices = [model.actuator("fl_joint7").id, model.actuator("fl_joint8").id]
left_body_id = model.body("fl_link6").id

def get_and_sync_state(model, data, joint_name_prefix, arm_indices, gripper_indices):
    state = np.zeros(8)
    try:
        for i in range(6):
            name = f"{joint_name_prefix}{i+1}"
            qadr = model.joint(name).qposadr[0]
            state[i] = data.qpos[qadr]
        g_name = f"{joint_name_prefix}7"
        state[6] = data.qpos[model.joint(g_name).qposadr[0]]
        state[7] = state[6]
        
        for i in range(6):
            data.ctrl[arm_indices[i]] = state[i]
        data.ctrl[gripper_indices[0]] = state[6]
        data.ctrl[gripper_indices[1]] = state[7]
    except Exception as e:
        state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01]
    return state, state[6]

print("Syncing state...")
right_arm_state, right_gripper_val = get_and_sync_state(model, data, "fr_joint", right_arm_actuator_ctrl_indices, right_gripper_indices)
left_arm_state, left_gripper_val = get_and_sync_state(model, data, "fl_joint", left_arm_actuator_ctrl_indices, left_gripper_indices)
print("Sync Complete.")

# Gripper Config
GRIPPER_MIN = ctrl_ranges[14][0]
GRIPPER_MAX = ctrl_ranges[14][1]
GRIPPER_STEP = 0.001 

# Initialize Controllers
rl_policy = BiActionChunkingTransformerGAT(num_joints=6, chunk_size=20)
fuzzy_controller_right = PredictiveTeleoperationController(num_joints=6, dt=0.002)
fuzzy_controller_left = PredictiveTeleoperationController(num_joints=6, dt=0.002)

# Visualization
visualizer = LearningVisualizer(history_len=200)
reporter = LearningReportGenerator(visualizer, rl_policy)
FUZZY_FORCE_GAIN = 5.0 

# =============================================================================
# 4. MAIN LOOP
# =============================================================================

if __name__ == "__main__":
    print("\n=== Haptic Dual Arm Control + HYBRID (RL + Fuzzy) ===")
  
    # Scaling Config
    SCALE_J1 = 1.0
    SCALE_J2 = 1.0
    SCALE_J3 = 1.0
    SCALE_J4 = 1.0
    SCALE_J5 = 1.0
    SCALE_J6 = 1.0   
    # Scaling Config
    #SCALE_J1 = 4.0
    #SCALE_J2 = 4.0
    #SCALE_J3 = 4.0
    #SCALE_J4 = 2.0
    #SCALE_J5 = 2.0
    #SCALE_J6 = 2.0
    
    #print(f"J1 to J3 Scale: 4.0x | J4 to J6 Scale: 2.0x")

    print("Press ESC to Exit and Generate Report")
    print("-" * 50)
    
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        haptic_device = HapticDevice(callback=state_callback, scheduler_type="async")
        time.sleep(0.2)
        print("Haptic Device Connected")
    except Exception as e:
        print(f"Haptic Device Error: {e}")
        sys.exit(1)

    try:
        tty.setcbreak(sys.stdin.fileno())
        cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)

        active_arm = "right"
        prev_haptic_joints = np.zeros(6) 

        with viewer.launch_passive(model, data) as v:
            running = True
            debug_counter = 0
            
            while running:
                # 1. Keyboard Input
                if select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char == 'r' or char == 'R':
                        active_arm = "right"
                        print(">>> Active Arm: RIGHT")
                    elif char == 'l' or char == 'L':
                        active_arm = "left"
                        print(">>> Active Arm: LEFT")
                    elif char == '\x1b':
                        running = False

                # 2. Haptic Input
                h_joints = np.array(device_state.full_joints) 
                h_btn_top = device_state.btn_top
                h_btn_bottom = device_state.btn_bottom

                # 3. RL Smoothing (Generate Clean Intent)
                raw_delta = h_joints - prev_haptic_joints
                
                temp_delta_3 = raw_delta[3]
                raw_delta[3] = raw_delta[4] 
                raw_delta[4] = temp_delta_3 

                # Run RL Policy
                refined_delta, intent, kalman, comp, safe = rl_policy.predict(raw_delta)
                
                # Integrate to get Target Position (q_des)
                target_pos_right = right_arm_state.copy()
                target_pos_left = left_arm_state.copy()

                if active_arm == "right":
                    target_pos_right[:6] += refined_delta * np.array([SCALE_J1, SCALE_J2, SCALE_J3, SCALE_J4, SCALE_J5, SCALE_J6])
                    
                    # Gripper
                    if h_btn_top: right_gripper_val -= GRIPPER_STEP
                    elif h_btn_bottom: right_gripper_val += GRIPPER_STEP
                    right_gripper_val = clamp(right_gripper_val, GRIPPER_MIN, GRIPPER_MAX)
                    target_pos_right[6] = right_gripper_val
                    target_pos_right[7] = right_gripper_val

                    # === HYBRID CONTROL STEP ===
                    q_curr = right_arm_state[:6]
                    q_des = target_pos_right[:6]
                    
                    if not hasattr(fuzzy_controller_right, 'prev_e'):
                        fuzzy_controller_right.prev_e = np.zeros(6)
                    e = q_des - q_curr
                    e_dot = (e - fuzzy_controller_right.prev_e) / 0.002 # dt
                    r_curr = e_dot + (fuzzy_controller_right.kappa @ e)
                    fuzzy_controller_right.prev_e = e
                    
                    fuzzy_u, f_est = fuzzy_controller_right.control_law(q_des, q_curr, r_curr, r_curr, is_master_side=False)
                    
                    # 3. Apply Controls to Mujoco
                    # A. Position Control (From RL Intent)
                    for i in range(6):
                        data.ctrl[right_arm_actuator_ctrl_indices[i]] = clamp(target_pos_right[i], ctrl_ranges[i][0], ctrl_ranges[i][1])
                    data.ctrl[right_gripper_indices[0]] = target_pos_right[6]
                    data.ctrl[right_gripper_indices[1]] = target_pos_right[7]
                    
                    # B. Force Control (From Fuzzy Output)
                    # FIX: Slice fuzzy_u to match (3,) shape
                    data.xfrc_applied[right_body_id, 3:6] = fuzzy_u[3:6] * FUZZY_FORCE_GAIN

                    # Visualization Update: Pass BOTH safe (Fig 26) and fuzzy_u (Fig 27)
                    visualizer.update(h_joints, target_pos_right, raw_delta, refined_delta, intent, kalman, comp, 
                                    safe,          # <--- Restores Old Safety Data
                                    fuzzy_u,       # <--- Adds New Fuzzy Data
                                    right_gripper_val, right_gripper_val, arm_name="right")
                    
                    render_cam = cam_right_id
                    cam_text = "RIGHT (Hybrid: RL+FUZZY)"

                elif active_arm == "left":
                    target_pos_left[:6] += refined_delta * np.array([SCALE_J1, SCALE_J2, SCALE_J3, SCALE_J4, SCALE_J5, SCALE_J6])
                    
                    if h_btn_top: left_gripper_val -= GRIPPER_STEP
                    elif h_btn_bottom: left_gripper_val += GRIPPER_STEP
                    left_gripper_val = clamp(left_gripper_val, GRIPPER_MIN, GRIPPER_MAX)
                    target_pos_left[6] = left_gripper_val
                    target_pos_left[7] = left_gripper_val

                    # === HYBRID CONTROL STEP ===
                    q_curr = left_arm_state[:6]
                    q_des = target_pos_left[:6]
                    
                    if not hasattr(fuzzy_controller_left, 'prev_e'):
                        fuzzy_controller_left.prev_e = np.zeros(6)
                    e = q_des - q_curr
                    e_dot = (e - fuzzy_controller_left.prev_e) / 0.002
                    r_curr = e_dot + (fuzzy_controller_left.kappa @ e)
                    fuzzy_controller_left.prev_e = e
                    
                    fuzzy_u, f_est = fuzzy_controller_left.control_law(q_des, q_curr, r_curr, r_curr, is_master_side=False)

                    for i in range(6):
                        data.ctrl[left_arm_actuator_ctrl_indices[i]] = clamp(target_pos_left[i], ctrl_ranges[i+8][0], ctrl_ranges[i+8][1])
                    data.ctrl[left_gripper_indices[0]] = target_pos_left[6]
                    data.ctrl[left_gripper_indices[1]] = target_pos_left[7]

                    # FIX: Slice fuzzy_u to match (3,) shape
                    data.xfrc_applied[left_body_id, 3:6] = fuzzy_u[3:6] * FUZZY_FORCE_GAIN

                    visualizer.update(h_joints, target_pos_left, raw_delta, refined_delta, intent, kalman, comp, safe, fuzzy_u, left_gripper_val, left_gripper_val, arm_name="left")

                    render_cam = cam_left_id
                    cam_text = "LEFT (Hybrid: RL+FUZZY)"

                # Update State Variables for next loop
                right_arm_state = target_pos_right.copy()
                left_arm_state = target_pos_left.copy()

                # 6. Render Vision
                if renderer is not None and render_cam != -1 and debug_counter % 2 == 0:
                    try:
                        renderer.update_scene(data, camera=render_cam)
                        pixels = renderer.render()
                        if pixels is not None and pixels.size > 0:
                            img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                            cv2.putText(img, cam_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow("Wrist Camera", img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                running = False
                    except:
                        pass

                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(model.opt.timestep)
                debug_counter += 1
                
                prev_haptic_joints = h_joints

        cv2.destroyAllWindows()
        if renderer: renderer.close()

    except Exception as e:
        print(f"\nRuntime Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if 'haptic_device' in locals():
            haptic_device.close()
        
        print("\nSimulation finished. Generating Learning Report...")
        reporter.generate_report()
        print("Done.")
