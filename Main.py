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

# --- NEW: Matplotlib for Learning Reports ---
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
            print(f"⚠ Could not find end-effector containing '{body_name_part}'. Cartesian Control disabled for this arm.")
            self.enabled = False
        else:
            self.enabled = True

    def get_ee_pos(self, data):
        if not self.enabled: return np.zeros(3)
        return data.xpos[self.body_id].copy()

    def resolve_ik(self, data, target_pos, max_iter=10, damping=0.05):
        if not self.enabled: return np.zeros(self.model.nv)

        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, jacp, None, self.body_id)
        curr_pos = data.xpos[self.body_id]
        error = target_pos - curr_pos
        
        error_mag = np.linalg.norm(error)
        step_scale = 0.01 
        if error_mag > step_scale:
            error = error / error_mag * step_scale

        delta_q = jacp.T @ error * 0.5
        return delta_q

import numpy as np
from collections import deque

# ==========================================
# === BI-DIRECTIONAL TRANSFORMER + GAT + META-LEARNING + KALMAN + COMPLIANCE ===
# ==========================================
class BiActionChunkingPolicy:
    def __init__(self, num_joints=6, chunk_size=20, smoothing_factor=0.15, dt=1.0):
        self.num_joints = num_joints
        self.chunk_size = chunk_size
        self.base_smoothing_factor = smoothing_factor
        self.dt = dt 
        
        # --- 1. TEMPORAL TRANSFORMER PARAMETERS (History) ---
        self.d_k = num_joints
        np.random.seed(42)
        # Temporal Query, Key, Value Weights
        self.W_q = np.eye(num_joints) * 0.5 + np.random.randn(num_joints, num_joints) * 0.1
        self.W_k = np.eye(num_joints) * 0.5 + np.random.randn(num_joints, num_joints) * 0.1
        self.W_v = np.eye(num_joints) * 0.5 + np.random.randn(num_joints, num_joints) * 0.1
        
        # CRITICAL FIX: Matrix Output for independent joint tracking
        self.W_val = np.eye(num_joints) * 0.5 + np.random.randn(num_joints, num_joints) * 0.1
        self.transformer_intent = np.zeros(num_joints)
        
        # --- NOVELTY 1: SPATIAL GAT PARAMETERS (Joint Coupling) ---
        # Projects joint features for graph attention
        self.W_gat = np.eye(num_joints) * 0.2 + np.random.randn(num_joints, num_joints) * 0.1
        # Attention vector for calculating edge weights (LeakyReLU logic simplified)
        self.a_gat = np.random.rand(num_joints * 2) 
        
        # --- 2. KALMAN FILTER PARAMETERS ---
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

        self.compliance_gain = 0.5 
        self.action_history = deque([np.zeros(num_joints)] * chunk_size, maxlen=chunk_size)
        self.last_output_action = np.zeros(num_joints)
        
        # --- NOVELTY 2: META-LEARNING PARAMETERS ---
        # 'meta_confidence': 0.0 (Raw Input) -> 1.0 (Trust Model)
        self.meta_confidence = 0.5 
        self.meta_lr = 0.05 # How fast we adapt our trust
        
        self.is_active = True 
        print("✓ RL Policy Module Loaded (Transformer + GAT + Meta-Learning + Kalman + Compliance)")

    def _scaled_dot_product_attention(self, Q, K, V):
        """Standard Temporal Attention (Time over Time)"""
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        attention_weights = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) + 1e-9)
        output = np.dot(attention_weights, V)
        return output, attention_weights

    def _graph_attention_layer(self, h):
        """
        NOVELTY 1: Graph Attention Network (GAT)
        Computes attention between joints (Spatial).
        Input: h (num_joints,) - Features of each joint
        Output: h_prime (num_joints,) - Context-aware joint features
        """
        # 1. Linear Projection
        # Wh: (num_joints, num_joints) -> dot with input h -> (num_joints,)
        # Actually, for attention we need pairwise interactions.
        # Let h be (N,).
        # We need interaction matrix: (N, N) where M[i,j] = LeakyReLU(a * [Wh_i || Wh_j])
        
        # Compute Wh for all nodes: (N,)
        Wh = np.dot(h, self.W_gat.T)
        
        # Create pairwise features for fully connected graph
        # Use broadcasting to create (N, N, 2N) matrix efficiently
        # Wh_i_col = Wh[:, None]  (N, 1)
        # Wh_j_row = Wh[None, :]  (1, N)
        # We want to concat Wh_i and Wh_j for every pair.
        # For simplicity in NumPy, we use dot product similarity for attention weights
        # instead of concatenation (which is heavier).
        
        # Attention Scores (Similarity matrix)
        # e_ij = Wh_i . Wh_j
        e_matrix = np.outer(Wh, Wh) # (N, N)
        
        # Apply LeakyReLU (simplified: just pass through or ReLU if we had negative bias)
        # Here we ensure non-negative via exp later.
        
        # Mask self-loops? No, self-attention is allowed.
        
        # Softmax over rows (Node i looks at all j)
        attention_weights = np.exp(e_matrix) / (np.sum(np.exp(e_matrix), axis=1, keepdims=True) + 1e-9)
        
        # Aggregate: h_prime[i] = sum(alpha_ij * Wh_j)
        h_prime = np.dot(attention_weights, Wh)
        
        # Residual connection (for stability)
        return h + h_prime

    def _meta_update(self, prediction_error):
        """
        NOVELTY 2: Meta-Learning Adaptation
        Adjusts 'meta_confidence' based on recent prediction error magnitude.
        If error is high -> Decrease confidence (Trust Raw).
        If error is low -> Increase confidence (Trust Model).
        """
        # Normalize error roughly
        error_mag = np.linalg.norm(prediction_error)
        
        # Update rule: Theta = Theta + lr * (Target - Current)
        # Target is 1.0 if error is small, 0.0 if error is large.
        target = np.exp(-error_mag * 50.0) # Decay function
        
        # Gradient step on meta parameter
        self.meta_confidence += self.meta_lr * (target - self.meta_confidence)
        
        # Clamp
        self.meta_confidence = np.clip(self.meta_confidence, 0.0, 1.0)

    def predict(self, raw_action):
        if not self.is_active: return raw_action, raw_action, raw_action, 0.5, np.zeros(6)
        
        # ==========================================
        # PHASE 1: TEMPORAL TRANSFORMER
        # ==========================================
        self.action_history.append(raw_action)
        history_arr = np.array(list(self.action_history))
        
        Q = np.dot(history_arr, self.W_q)
        K = np.dot(history_arr, self.W_k)
        V = np.dot(history_arr, self.W_v)
        
        transformer_output, attn_weights = self._scaled_dot_product_attention(Q, K, V)
        
        # Take the latest temporal intent
        temporal_intent = transformer_output[-1] # Shape (6,)
        self.attention_weights = attn_weights[-1] 
        
        # Calculate value for context (noise estimation)
        trajectory_values = np.dot(transformer_output, self.W_val)
        current_rl_value = np.mean(np.abs(trajectory_values[-5:]))
        
        # ==========================================
        # PHASE 2: SPATIAL GAT (Joint Coupling)
        # ==========================================
        # Apply GAT to the temporal intent to capture joint dependencies
        gat_intent = self._graph_attention_layer(temporal_intent)
        
        # ==========================================
        # PHASE 3: META-LEARNING (Adaptive Trust)
        # ==========================================
        # Calculate error between GAT intent and Raw Input to gauge model performance
        instant_error = gat_intent - raw_action
        self._meta_update(instant_error)
        
        # Meta-adapted Intent: Blend GAT output with Raw Input based on Meta Confidence
        # If confidence is 1.0 -> Use GAT. If 0.0 -> Use Raw.
        meta_adapted_intent = (self.meta_confidence * gat_intent) + \
                              ((1.0 - self.meta_confidence) * raw_action)

        # ==========================================
        # PHASE 4: KALMAN FILTER (Smoothing)
        # ==========================================
        u = np.expand_dims(meta_adapted_intent - self.last_output_action, axis=0) 
        
        # Meta-Learning also adjusts Kalman R (Measurement Noise)
        # Higher confidence -> Lower R -> Trust Model (Smooth)
        # Lower confidence -> Higher R -> Trust Measurement (Reactive)
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

        # ==========================================
        # PHASE 5: COMPLIANCE
        # ==========================================
        current_vel = self.x_hat[1, :] 
        speed = np.linalg.norm(current_vel)
        max_speed_threshold = 0.05
        adaptive_compliance = np.clip(1.0 - (speed / max_speed_threshold), 0.1, 0.9)
        
        final_action = (adaptive_compliance * corrected_pos) + \
                       ((1.0 - adaptive_compliance) * meta_adapted_intent)

        # ==========================================
        # PHASE 6: SAFETY
        # ==========================================
        delta = final_action - self.last_output_action
        max_accel = 0.005 
        safety_clamp = delta 
        delta_clamped = np.clip(delta, -max_accel, max_accel)
        final_action = self.last_output_action + delta_clamped
        
        self.last_output_action = final_action
        
        # Store the Meta-Adapted intent for visualization (shows the "Coupled" intent)
        self.transformer_intent = meta_adapted_intent
        
        return (final_action, self.transformer_intent, corrected_pos, adaptive_compliance, safety_clamp)


# ==========================================
# === LEARNING VISUALIZER (DATA COLLECTOR) ===
# ==========================================
class LearningVisualizer:
    def __init__(self, history_len=200):
        self.history_len = history_len
        
        # 1. POSITION TRACKING
        self.right_h_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_h_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_r_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_r_pos_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        
        # 2. VELOCITY TRACKING
        self.right_h_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_h_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_s_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.left_s_vel_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]

        # 3. LEGACY VARIABLES (Figures 23, 24 need specific vector structures)
        # We keep these for compatibility with existing plots that expect single deques
        self.transformer_intent_hist = deque([np.zeros(3) for _ in range(history_len)], maxlen=history_len)
        self.kalman_corrected_hist = deque([np.zeros(3) for _ in range(history_len)], maxlen=history_len)
        self.compliance_factor_hist = deque([0.0]*history_len, maxlen=history_len)
        
        # UPDATED: Split Safety Clamp into Left and Right 6-DOF lists for Figure 26
        self.left_safe_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]
        self.right_safe_hist = [deque([0.0]*history_len, maxlen=history_len) for _ in range(6)]

        # Grippers
        self.right_j7_hist = deque([0.0]*history_len, maxlen=history_len)
        self.right_j8_hist = deque([0.0]*history_len, maxlen=history_len)
        self.left_j7_hist = deque([0.0]*history_len, maxlen=history_len)
        self.left_j8_hist = deque([0.0]*history_len, maxlen=history_len)

    def update(self, haptic_pos, robot_pos, haptic_vel, smooth_vel, 
               transformer_intent, kalman_corrected, compliance_factor, safety_clamp, j7, j8, arm_name="right"):
        
        # Store Position
        if arm_name == "right":
            for i in range(6):
                self.right_h_pos_hist[i].append(haptic_pos[i])
                self.right_r_pos_hist[i].append(robot_pos[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_h_pos_hist[i].append(haptic_pos[i])
                self.left_r_pos_hist[i].append(robot_pos[i])

        # Store Velocities
        if arm_name == "right":
            for i in range(6):
                self.right_h_vel_hist[i].append(haptic_vel[i])
                self.right_s_vel_hist[i].append(smooth_vel[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_h_vel_hist[i].append(haptic_vel[i])
                self.left_s_vel_hist[i].append(smooth_vel[i])

        # Store Grippers
        if arm_name == "right":
            self.right_j7_hist.append(j7)
            self.right_j8_hist.append(j8)
        elif arm_name == "left":
            self.left_j7_hist.append(j7)
            self.left_j8_hist.append(j8)

        # Policy Internals (Legacy 3-DOF storage)
        self.transformer_intent_hist.append(transformer_intent[:3])
        self.kalman_corrected_hist.append(kalman_corrected[:3])
        self.compliance_factor_hist.append(compliance_factor)

        # UPDATED: Store Safety Clamp (6-DOF)
        if arm_name == "right":
            for i in range(6):
                self.right_safe_hist[i].append(safety_clamp[i])
        elif arm_name == "left":
            for i in range(6):
                self.left_safe_hist[i].append(safety_clamp[i])

# ==========================================
# === LEARNING REPORT GENERATOR (PDF) ===
# ==========================================
class LearningReportGenerator:
    def __init__(self, visualizer, policy): 
        self.viz = visualizer
        self.policy = policy
        save_dir = os.path.expanduser("~/Desktop")
        self.pdf_filename = os.path.join(save_dir, "ALOHA-Haptic-Control-Report.pdf")

    def generate_report(self):
        print(f"\nGenerating Learning Report (PDF): {self.pdf_filename}")
        print("Please close pop-up windows to continue...")
        
        # --- DATA EXTRACTION ---
        # Using Joint 3 (Index 2) as representative for scalar plots (Fig 1, 2, 8, etc.)
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
        ax1.set_title("Figure 1: Bi-Action Smoothing Analysis (Overall 6-DOF)", fontsize=14)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Overall Velocity Magnitude (L2 Norm)")
        
        try:
            raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            
            raw_overall = np.linalg.norm(raw_vel_matrix, axis=0)
            smooth_overall = np.linalg.norm(smooth_vel_matrix, axis=0)
        except (AttributeError, IndexError) as e:
            print(f"Warning: Visualization structure mismatch in Fig 1: {e}")
            raw_overall = np.zeros(len(steps))
            smooth_overall = np.zeros(len(steps))

        if len(raw_overall) > 0:
            var_raw = np.var(raw_overall)
            var_smooth = np.var(smooth_overall)
            avg_raw = np.mean(raw_overall)
            avg_smooth = np.mean(smooth_overall)
            
            reduction_pct = ((var_raw - var_smooth) / var_raw * 100) if var_raw > 1e-9 else 0.0

            raw_label = f"Raw Input (Avg: {avg_raw:.3f})"
            smooth_label = f"Smoothed Output (Avg: {avg_smooth:.3f}, Overall Tracking Error Reduction: {reduction_pct:.1f}%)"

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
            ax2.set_title("Figure 2: Smoothness Metric (Variance) over Time", fontsize=14)
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Signal Variance (Lower is Smoother)")
            
            ax2.plot(time_window, variance_raw, label=f"Raw Input (Avg Variance: {avg_var_raw:.4f})", color='red', alpha=0.6, linestyle='--')
            ax2.plot(time_window, variance_smooth, label=f"Smoothed Output (Avg Variance: {avg_var_smooth:.4f}, Reduction of Variance: {reduction_pct:.1f}%)", color='purple', linewidth=2)
            
            ax2.fill_between(time_window, variance_smooth, 0, color='purple', alpha=0.1)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            plt.savefig(pp, format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()
            
        # --- FIGURE 3 (UPDATED: Average of Left and Right Arms) ---
        print("Generating Figure 3: Joint Trajectories (Average Left + Right Arms)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.set_title("Figure 3: Joint Trajectories (Average of Left & Right Arms)", fontsize=14)
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

            raw_label = f"Avg Raw Input (L+R) (Avg: {avg_haptic_val:.3f})"
            robot_label = f"Avg Smoothed Robot (L+R) (Avg: {avg_robot_val:.3f}, Error reduction MAE: {pct_change:.1f}%)"

            ax3.plot(steps[:len(avg_raw_overall)], avg_raw_overall, label=raw_label, color='gray', linestyle='--', alpha=0.6)
            ax3.plot(steps[:len(avg_robot_overall)], avg_robot_overall, label=robot_label, color='blue', linewidth=2)
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            plt.savefig(pp, format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()
        except (AttributeError, IndexError) as e:
            print(f"Warning: Visualization structure mismatch in Fig 3: {e}")


        # --- FIGURE 4 (KEPT AS REQUESTED) ---
        print("Generating Figure 4: Joint Trajectories (Both Arms - 6-Joint)")
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
        fig4.suptitle("Figure 4: Joint Trajectories (Overall 6-DOF)", fontsize=16)
        
        try:
            r_robot_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            r_haptic_matrix = np.array([list(self.viz.right_h_pos_hist[i]) for i in range(6)])
        except AttributeError:
            print("Warning: Right Arm history not found in Viz.")
            r_robot_matrix = np.zeros((6, len(steps)))
            r_haptic_matrix = np.zeros((6, len(steps)))
            
        r_robot_overall = np.linalg.norm(r_robot_matrix, axis=0)
        r_haptic_overall = np.linalg.norm(r_haptic_matrix, axis=0)
        
        r_avg_robot = np.mean(r_robot_overall)
        r_mae = np.mean(np.abs(r_haptic_overall - r_robot_overall))
        r_avg_haptic = np.mean(r_haptic_overall) 
        r_pct = (r_mae / r_avg_haptic * 100) if r_avg_haptic > 1e-9 else 0.0
        
        ax4a.set_title("Right Arm (6-DOF)")
        ax4a.plot(steps[:len(r_haptic_overall)], r_haptic_overall, 
                label=f"Haptic (Avg: {r_avg_haptic:.4f})", color='gray', linestyle='--', alpha=0.6)
        ax4a.plot(steps[:len(r_robot_overall)], r_robot_overall, 
                label=f"Robot (Avg: {r_avg_robot:.4f}, Error reduction: {r_pct:.4f}%)", color='green', linewidth=2)
        ax4a.legend(loc='upper right')
        ax4a.grid(True, alpha=0.3)

        try:
            l_robot_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            l_haptic_matrix = np.array([list(self.viz.left_h_pos_hist[i]) for i in range(6)])
        except AttributeError:
            print("Warning: Left Arm history not found in Viz.")
            l_robot_matrix = np.zeros((6, len(steps)))
            l_haptic_matrix = np.zeros((6, len(steps)))

        l_robot_overall = np.linalg.norm(l_robot_matrix, axis=0)
        l_haptic_overall = np.linalg.norm(l_haptic_matrix, axis=0)
        
        l_avg_robot = np.mean(l_robot_overall)
        l_mae = np.mean(np.abs(l_haptic_overall - l_robot_overall))
        l_avg_haptic = np.mean(l_haptic_overall)
        l_pct = (l_mae / l_avg_haptic * 100) if l_avg_haptic > 1e-9 else 0.0

        ax4b.set_title("Left Arm (6-DOF)")
        ax4b.plot(steps[:len(l_haptic_overall)], l_haptic_overall, 
                label=f"Haptic (Avg: {l_avg_haptic:.4f})", color='gray', linestyle='--', alpha=0.6)
        ax4b.plot(steps[:len(l_robot_overall)], l_robot_overall, 
                label=f"Robot (Avg: {l_avg_robot:.4f}, MAE Error reduction: {l_pct:.4f}%)", color='purple', linewidth=2)
        ax4b.legend(loc='upper right')
        ax4b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        

        # --- FIGURE 5 (UPDATED: Overall Velocity) ---
        print("Generating Figure 5: Action Residuals (Overall 6-DOF) with Detailed Metrics")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.set_title("Figure 5: Action Residuals (Overall 6-DOF) with Metrics", fontsize=14)
        ax5.set_xlabel("Time Steps")
        ax5.set_ylabel("Overall Velocity Magnitude (L2 Norm)")
        
        try:
            raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            velocity_raw = np.linalg.norm(raw_vel_matrix, axis=0)
            velocity_smooth = np.linalg.norm(smooth_vel_matrix, axis=0)
        except AttributeError:
            print("Warning: Visualization structure mismatch in Fig 5.")
            velocity_raw = np.zeros(len(steps))
            velocity_smooth = np.zeros(len(steps))

        residuals = velocity_raw - velocity_smooth
        
        avg_res = np.mean(np.abs(residuals))
        mean_raw_vel = np.mean(np.abs(velocity_raw))
        mean_smooth_vel = np.mean(np.abs(velocity_smooth))
        var_raw = np.var(velocity_raw)
        var_smooth = np.var(velocity_smooth)
        
        reduction_pct = ((var_raw - var_smooth) / var_raw * 100) if var_raw > 1e-9 else 0.0
        
        if mean_raw_vel > 1e-9:
            residual_reduction = ((mean_raw_vel - avg_res) / mean_raw_vel) * 100
        else:
            residual_reduction = 0.0

        label_raw = f"Raw Input (Mean: {mean_raw_vel:.4f})"
        label_smooth = f"Smoothed Output (Mean: {mean_smooth_vel:.4f}, MAE Var Reduction: {reduction_pct:.1f}%)"
        label_residual = f"Residuals (Mean: {avg_res:.3f}, SNR Error Reduction: {residual_reduction:.1f}%)"

        ax5.plot(steps, velocity_raw, label=label_raw, color='red', alpha=0.5, linestyle='--')
        ax5.plot(steps, velocity_smooth, label=label_smooth, color='green', linewidth=2)
        ax5.plot(steps, residuals, label=label_residual, color='magenta', linewidth=1.5, linestyle='-')
        ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax5.fill_between(steps, residuals, 0, color='magenta', alpha=0.1)
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


        # --- FIGURE 6 (UPDATED: 12 Subplots - Each Joint Separately) ---
        print("Generating Figure 6: Frequency Spectrum (Left & Right Arms - Individual Joints)")
        
        fig6, axes = plt.subplots(2, 6, figsize=(20, 10))
        fig6.suptitle("Figure 6: Frequency Spectrum (Per Joint Analysis)", fontsize=16)
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
        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 6: {e}")
            left_data = [np.zeros(len(steps)) for _ in range(6)]
            right_data = [np.zeros(len(steps)) for _ in range(6)]

        for i in range(6):
            plot_spectrogram(axes[0, i], left_data[i], f"Left Arm J{i+1}", 'inferno')
            plot_spectrogram(axes[1, i], right_data[i], f"Right Arm J{i+1}", 'viridis')

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        

        # --- FIGURE 7 (UPDATED: 3 Subplots - Left, Right, Avg) ---
        print("Generating Figure 7: Jerk Profile (Left, Right & Average - All 6-DOF)")
        
        from matplotlib.gridspec import GridSpec

        fig7 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig7)

        ax7a = fig7.add_subplot(gs[0, 0]) 
        ax7b = fig7.add_subplot(gs[0, 1]) 
        ax7c = fig7.add_subplot(gs[1, :]) 
        
        fig7.suptitle("Figure 7: Jerk Profile Analysis (6-DOF L2 Norm)", fontsize=16)

        def calculate_overall_jerk(pos_matrix):
            vel = np.gradient(pos_matrix, axis=1)
            acc = np.gradient(vel, axis=1)
            jerk = np.gradient(acc, axis=1)
            return np.linalg.norm(jerk, axis=0)

        def plot_jerk_subplot(ax, raw_jerk, smooth_jerk, title, color_raw, color_smooth):
            avg_raw = np.mean(raw_jerk)
            avg_smooth = np.mean(smooth_jerk)
            
            if avg_raw > 1e-9:
                improvement = ((avg_raw - avg_smooth) / avg_raw) * 100
            else:
                improvement = 0.0

            ax.plot(steps[:len(raw_jerk)], raw_jerk, 
                   label=f'Raw (Avg: {avg_raw:.3f})', color=color_raw, alpha=0.4, linestyle='--')
            ax.plot(steps[:len(smooth_jerk)], smooth_jerk, 
                   label=f'Smooth (Avg: {avg_smooth:.3f}, Imp: {improvement:.1f}%)', color=color_smooth, linewidth=1.5)
            ax.set_title(title)
            ax.set_ylabel("Jerk Magnitude")
            ax.legend(loc='upper right', fontsize='small')
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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 7: {e}")
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
        if avg_r > 1e-9:
            improvement = ((avg_r - avg_s) / avg_r) * 100
        else:
            improvement = 0.0

        ax7c.plot(steps[:len(avg_raw_jerk)], avg_raw_jerk, 
               label=f'Avg Raw (L+R) (Avg: {avg_r:.3f})', color='purple', alpha=0.4, linestyle='--')
        ax7c.plot(steps[:len(avg_smooth_jerk)], avg_smooth_jerk, 
               label=f'Avg Smooth (L+R) (Avg: {avg_s:.3f}, Imp: {improvement:.1f}%)', color='green', linewidth=1.5)
        ax7c.set_title("Average Jerk (Left + Right Arms)")
        ax7c.set_xlabel("Time Steps")
        ax7c.set_ylabel("Average Jerk Magnitude")
        ax7c.legend(loc='upper right')
        ax7c.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # --- FIGURE 8 (Latency) ---
        print("Generating Figure 8: Input-Output Latency (Lag Analysis)")
        raw_norm = (raw_vel_3 - np.mean(raw_vel_3)) / (np.std(raw_vel_3) + 1e-8)
        smooth_norm = (smooth_vel_3 - np.mean(smooth_vel_3)) / (np.std(smooth_vel_3) + 1e-8)
        correlation = np.correlate(raw_norm, smooth_norm, mode='full')
        lags = np.arange(-len(raw_norm) + 1, len(raw_norm))
        peak_idx = np.argmax(correlation)
        peak_lag = lags[peak_idx]
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.set_title(f"Figure 8: Input-Output Latency (Lag: {peak_lag} steps)", fontsize=14)
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

        # --- FIGURE 9 (3D Trajectory) ---
        print("Generating Figure 9: 3D Joint Trajectory Space")
        fig9 = plt.figure(figsize=(10, 8))
        ax9 = fig9.add_subplot(111, projection='3d')
        ax9.set_title("Figure 9: 3D Joint Trajectory Space (Robot)", fontsize=14)
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
        ax10.set_title("Figure 10: Policy Attention Mechanism (History Weights)", fontsize=14)
        ax10.set_xlabel("History Steps (0 = Oldest, Right = Current)")
        ax10.set_ylabel("Attention Weight (Importance)")
        weights = self.policy.attention_weights
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

        # --- FIGURE 11 (UPDATED: Overall 6-DOF Error Distribution) ---
        print("Generating Figure 11: Statistical Distribution of Filtering Errors (6-DOF)")
        
        try:
            raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            overall_raw_vel = np.linalg.norm(raw_vel_matrix, axis=0)
            overall_smooth_vel = np.linalg.norm(smooth_vel_matrix, axis=0)
        except AttributeError:
            print("Warning: Visualization structure mismatch in Fig 11.")
            overall_raw_vel = np.zeros(len(steps))
            overall_smooth_vel = np.zeros(len(steps))

        residuals = overall_raw_vel - overall_smooth_vel

        fig11, ax11 = plt.subplots(figsize=(10, 6))
        ax11.set_title("Figure 11: Statistical Distribution of Filtering Errors (6-DOF)", fontsize=14)
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
            ax11.plot(bins, best_fit_line * len(residuals) * (bins[1]-bins[0]), 'r--', linewidth=2, 
                      label=rf'Normal Fit ($\mu={mu:.3f}, \sigma={sigma:.3f}$)')
        
        ax11.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax11.legend(loc='upper right')
        ax11.grid(True, alpha=0.3)
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()



        # --- FIGURE 12 (UPDATED: 3 Subplots - Left, Right, Avg) ---
        print("Generating Figure 12: Frequency Attenuation (Left, Right & Average - All 6-DOF)")
        
        from matplotlib.gridspec import GridSpec

        fig12 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig12)

        ax12a = fig12.add_subplot(gs[0, 0]) 
        ax12b = fig12.add_subplot(gs[0, 1]) 
        ax12c = fig12.add_subplot(gs[1, :]) 
        
        fig12.suptitle("Figure 12: Frequency Attenuation (6-DOF L2 Norm)", fontsize=16)

        def plot_psd_subplot(ax, raw_vel, smooth_vel, title, label_prefix):
            freqs_raw, psd_raw = scipy.signal.welch(raw_vel, fs=60.0, nperseg=64)
            freqs_smooth, psd_smooth = scipy.signal.welch(smooth_vel, fs=60.0, nperseg=64)

            psd_raw_safe = np.maximum(psd_raw, 1e-20)
            psd_smooth_safe = np.maximum(psd_smooth, 1e-20)

            avg_psd_raw = np.mean(psd_raw_safe)
            avg_psd_smooth = np.mean(psd_smooth_safe)

            if avg_psd_raw > 1e-9:
                reduction_pct = ((avg_psd_raw - avg_psd_smooth) / avg_psd_raw) * 100
            else:
                reduction_pct = 0.0

            ax.semilogy(freqs_raw, psd_raw_safe, label=f'{label_prefix} Raw (Avg PSD: {avg_psd_raw:.7f})', color='red', alpha=0.6)
            ax.semilogy(freqs_smooth, psd_smooth_safe, 
                       label=f'{label_prefix} Smooth (Avg PSD: {avg_psd_smooth:.7f}, Reduced: {reduction_pct:.1f}%)', 
                       color='blue', linewidth=2)
            ax.set_title(title)
            ax.set_ylabel("Power Spectral Density (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlim([0, 30])

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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 12: {e}")
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

        
        # --- FIGURE 13 (UPDATED: 3 Subplots - Left, Right, Avg) ---
        print("Generating Figure 13: Velocity Profile (Left, Right & Average - All 6-DOF)")
        
        fig13 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig13)

        ax13a = fig13.add_subplot(gs[0, 0]) 
        ax13b = fig13.add_subplot(gs[0, 1]) 
        ax13c = fig13.add_subplot(gs[1, :]) 
        
        fig13.suptitle("Figure 13: Velocity Profile Analysis (6-DOF L2 Norm)", fontsize=16)

        def plot_velocity_subplot(ax, raw_vel, smooth_vel, title, label_prefix):
            avg_raw = np.mean(raw_vel)
            avg_smooth = np.mean(smooth_vel)
            
            if avg_raw > 1e-9:
                pct_change = ((avg_raw - avg_smooth) / avg_raw) * 100
            else:
                pct_change = 0.0

            ax.plot(steps[:len(raw_vel)], raw_vel, 
                   label=f'{label_prefix} Raw (Avg: {avg_raw:.4f})', color='red', alpha=0.3, linewidth=1)
            ax.plot(steps[:len(smooth_vel)], smooth_vel, 
                   label=f'{label_prefix} Smooth (Avg: {avg_smooth:.4f}, Chg: {pct_change:.1f}%)', color='green', linewidth=1.5)
            ax.fill_between(steps[:len(smooth_vel)], smooth_vel, 0, color='green', alpha=0.1)
            ax.set_title(title)
            ax.set_ylabel("Overall Velocity Magnitude")
            ax.legend(loc='upper right', fontsize='small')
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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 13: {e}")
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
        if avg_r > 1e-9:
            pct_change = ((avg_r - avg_s) / avg_r) * 100
        else:
            pct_change = 0.0

        ax13c.plot(steps[:len(avg_vel_norm)], avg_vel_norm, 
               label=f'Avg Raw (L+R) (Avg: {avg_r:.4f})', color='purple', alpha=0.3, linewidth=1)
        ax13c.plot(steps[:len(avg_smooth_vel_norm)], avg_smooth_vel_norm, 
               label=f'Avg Smooth (L+R) (Avg: {avg_s:.4f}, Chg: {pct_change:.1f}%)', color='blue', linewidth=1.5)
        ax13c.fill_between(steps[:len(avg_smooth_vel_norm)], avg_smooth_vel_norm, 0, color='blue', alpha=0.1)
        ax13c.set_title("Average Velocity (Left + Right Arms)")
        ax13c.set_xlabel("Time Steps")
        ax13c.set_ylabel("Average Velocity Magnitude")
        ax13c.legend(loc='upper right')
        ax13c.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
        
        # --- FIGURE 14 (UPDATED: 3 Subplots with Average Values in Legend) ---
        print("Generating Figure 14: Input-Output Hysteresis (Left, Right & Avg - All 6-DOF)")
        
        fig14 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig14)

        ax14a = fig14.add_subplot(gs[0, 0]) 
        ax14b = fig14.add_subplot(gs[0, 1]) 
        ax14c = fig14.add_subplot(gs[1, :]) 
        
        fig14.suptitle("Figure 14: Input-Output Hysteresis-changes of lags (6-DOF L2 Norm)", fontsize=16)

        def plot_hysteresis_subplot(ax, haptic_matrix, robot_matrix, title, label_prefix):
            haptic_norm = np.linalg.norm(haptic_matrix, axis=0)
            robot_norm = np.linalg.norm(robot_matrix, axis=0)
            
            mean_haptic_val = np.mean(haptic_norm)
            mean_robot_val = np.mean(robot_norm)
            
            limit = max(np.max(haptic_norm), np.max(robot_norm)) * 1.1
            limit = max(limit, 0.1)

            ax.plot([0, limit], [0, limit], 'k--', alpha=0.5, 
                   label=f'Ideal Tracking ({label_prefix} Raw: {mean_haptic_val:.4f}, Robot: {mean_robot_val:.4f})')
            
            sc = ax.scatter(haptic_norm, robot_norm, c=steps, cmap='plasma', s=10, alpha=0.6)
            
            ax.set_title(title)
            ax.set_xlabel(f"{label_prefix} Haptic Pos (L2 Norm)")
            ax.set_ylabel(f"{label_prefix} Robot Pos (L2 Norm)")
            ax.set_xlim([0, limit])
            ax.set_ylim([0, limit])
            ax.grid(True, alpha=0.3)
            return sc

        try:
            l_haptic_matrix = np.array([list(self.viz.left_h_pos_hist[i]) for i in range(6)])
            l_robot_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_haptic_matrix = np.array([list(self.viz.right_h_pos_hist[i]) for i in range(6)])
            r_robot_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            avg_haptic_norm = (np.linalg.norm(l_haptic_matrix, axis=0) + np.linalg.norm(r_haptic_matrix, axis=0)) / 2.0
            avg_robot_norm = (np.linalg.norm(l_robot_matrix, axis=0) + np.linalg.norm(r_robot_matrix, axis=0)) / 2.0

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 14: {e}")
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
        
        ax14c.plot([0, limit_avg], [0, limit_avg], 'k--', alpha=0.5, 
                   label=f'Ideal Tracking (Avg Raw: {mean_haptic_avg:.4f}, Avg Robot: {mean_robot_avg:.4f})')
        sc_avg = ax14c.scatter(avg_haptic_norm, avg_robot_norm, c=steps, cmap='plasma', s=10, alpha=0.6)
        ax14c.set_title("Average Position (Left + Right Arms)")
        ax14c.set_xlabel("Avg Haptic Pos (L2 Norm)")
        ax14c.set_ylabel("Avg Robot Pos (L2 Norm)")
        ax14c.set_xlim([0, limit_avg])
        ax14c.set_ylim([0, limit_avg])
        ax14c.grid(True, alpha=0.3)

        fig14.colorbar(sc_avg, ax=ax14c, label='Time Step')
        
        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


        # --- FIGURE 15 (UPDATED: 3 Subplots - Left, Right, Avg) ---
        print("Generating Figure 15: Cumulative Control Effort (Left, Right & Avg - All 6-DOF)")
        
        fig15 = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig15)

        ax15a = fig15.add_subplot(gs[0, 0]) 
        ax15b = fig15.add_subplot(gs[0, 1]) 
        ax15c = fig15.add_subplot(gs[1, :]) 
        
        fig15.suptitle("Figure 15: Cumulative Control Effort (6-DOF L2 Norm)", fontsize=16)

        def plot_effort_subplot(ax, raw_vel_matrix, smooth_vel_matrix, title, label_prefix):
            raw_vel = np.linalg.norm(raw_vel_matrix, axis=0)
            smooth_vel = np.linalg.norm(smooth_vel_matrix, axis=0)
            
            effort_raw = np.cumsum(raw_vel)
            effort_smooth = np.cumsum(smooth_vel)
            
            avg_raw = np.mean(raw_vel)
            avg_smooth = np.mean(smooth_vel)

            ax.plot(steps[:len(raw_vel)], effort_raw, 
                   label=f'{label_prefix} Raw (Avg Vel: {avg_raw:.4f})', color='red', alpha=0.6, linestyle='--')
            ax.plot(steps[:len(smooth_vel)], effort_smooth, 
                   label=f'{label_prefix} Smooth (Avg Vel: {avg_smooth:.4f})', color='blue', linewidth=2)
            
            final_diff = effort_raw[-1] - effort_smooth[-1]
            ax.fill_between(steps[:len(effort_smooth)], effort_smooth, effort_raw, color='red', alpha=0.1, 
                         label=f'Energy Saved: {final_diff:.2f}')
            
            ax.set_title(title)
            ax.set_ylabel("Accumulated Effort (Arbitrary Units)")
            ax.legend(loc='upper left', fontsize='small')
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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 15: {e}")
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

        ax15c.plot(steps[:len(avg_vel_norm)], effort_avg_raw, 
               label=f'Avg Raw (L+R) (Avg Vel: {avg_r:.4f})', color='purple', alpha=0.6, linestyle='--')
        ax15c.plot(steps[:len(avg_smooth_vel_norm)], effort_avg_smooth, 
               label=f'Avg Smooth (L+R) (Avg Vel: {avg_s:.4f})', color='green', linewidth=2)
        ax15c.fill_between(steps[:len(avg_smooth_vel_norm)], effort_avg_smooth, effort_avg_raw, color='red', alpha=0.1, 
                     label=f'Energy Saved: {final_diff:.2f}')
        ax15c.set_title("Average Cumulative Effort (Left + Right Arms)")
        ax15c.set_ylabel("Accumulated Effort (Arbitrary Units)")
        ax15c.set_xlabel("Time Steps")
        ax15c.legend(loc='upper left')
        ax15c.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


        # --- FIGURE 16 (UPDATED: 3 Subplots in One Row - Left, Right, Avg) ---
        print("Generating Figure 16: Motion Coupling (6-DOF Correlation Matrix)")

        fig16, (ax16a, ax16b, ax16c) = plt.subplots(1, 3, figsize=(18, 6))
        fig16.suptitle("Figure 16: Motion Coupling (6-DOF Correlation Matrix)", fontsize=16)

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
            
            ax.set_title(title)

        try:
            l_pos_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_pos_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            avg_pos_matrix = (l_pos_matrix + r_pos_matrix) / 2.0

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 16: {e}")
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

        # --- FIGURE 17 (UPDATED: Left, Right, Avg Scalograms - All 6-DOF) ---
        print("Generating Figure 17: Scalogram (Time-Frequency Intensity - 6-DOF)")
        
        fig17, (ax17a, ax17b, ax17c) = plt.subplots(1, 3, figsize=(18, 6))
        fig17.suptitle("Figure 17: Scalogram (6-DOF L2 Norm Analysis)", fontsize=16)

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
            ax.set_title(title)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Wavelet Scale (Width)")

        try:
            l_pos_matrix = np.array([list(self.viz.left_r_pos_hist[i]) for i in range(6)])
            r_pos_matrix = np.array([list(self.viz.right_r_pos_hist[i]) for i in range(6)])
            l_overall_pos = np.linalg.norm(l_pos_matrix, axis=0)
            r_overall_pos = np.linalg.norm(r_pos_matrix, axis=0)
            avg_overall_pos = (l_overall_pos + r_overall_pos) / 2.0

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 17: {e}")
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


        # --- FIGURE 18 (FIXED: Array Dimension Mismatch) ---
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
            
            ax.set_title(title)
            ax.set_xlabel("Time Lags")
            ax.set_ylabel("Correlation Coefficient")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 50])

        fig18, (ax18a, ax18b, ax18c) = plt.subplots(1, 3, figsize=(18, 6))
        fig18.suptitle("Figure 18: Autocorrelation Function (6-DOF L2 Norm)", fontsize=16)

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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 18: {e}")
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
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
                
        # --- FIGURE 19 (UPDATED: 3 Subplots in One Row - Left, Right, Avg) ---
        print("Generating Figure 19: System Performance Quantification (6-DOF L2 Norm)")

        def calculate_performance_metrics(raw_vel, smooth_vel):
            rmse = np.sqrt(np.mean((raw_vel - smooth_vel)**2))
            mae = np.mean(np.abs(raw_vel - smooth_vel))
            p2p_raw = np.max(raw_vel) - np.min(raw_vel)
            p2p_smooth = np.max(smooth_vel) - np.min(smooth_vel)
            
            if p2p_raw > 1e-9:
                p2p_reduction = (1 - p2p_smooth / p2p_raw) * 100
            else:
                p2p_reduction = 0.0
            
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
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, max(100, np.max(values)*1.1)])

        fig19, (ax19a, ax19b, ax19c) = plt.subplots(1, 3, figsize=(18, 6))
        fig19.suptitle("Figure 19: System Performance Quantification (6-DOF L2 Norm)", fontsize=16)

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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 19: {e}")
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
        ax20.set_title("Figure 20: Magnitude Coherence (Frequency Tracking Fidelity)", fontsize=14)
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

        # --- FIGURE 21 (UPDATED: 3 Subplots in One Row - Left, Right, Avg) ---
        print("Generating Figure 21: Error Dynamics (Stability Analysis - 6-DOF L2 Norm)")

        fig21, (ax21a, ax21b, ax21c) = plt.subplots(1, 3, figsize=(18, 6))
        fig21.suptitle("Figure 21: Error Dynamics (6-DOF L2 Norm)", fontsize=16)

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

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 21: {e}")
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

        # --- FIGURE 22 (UPDATED: 3 Subplots - Left, Right, Avg - 6-DOF) ---
        print("Generating Figure 22: Shannon Entropy (6-DOF Information Content): Less entropy indicates stable , high indicates Shaky or chaotic")

        from matplotlib.gridspec import GridSpec

        fig22, (ax22a, ax22b, ax22c) = plt.subplots(1, 3, figsize=(18, 6))
        fig22.suptitle("Figure 22: Shannon Entropy (6-DOF L2 Norm)", fontsize=16)

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
            
            ax.plot(time_indices, entropy_raw, 
                   label=f'{label_prefix} Raw (Avg: {avg_ent_r:.3f})', color='red', alpha=0.6, linewidth=1)
            ax.plot(time_indices, entropy_smooth, 
                   label=f'{label_prefix} Smooth (Avg: {avg_ent_s:.3f})', color='blue', linewidth=2)
            ax.fill_between(time_indices, entropy_smooth, entropy_raw, color='green', alpha=0.1, label='Redundant Noise Removed')
            ax.set_title(title)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Entropy (Bits)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            return entropy_raw, entropy_smooth

        try:
            l_h_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_s_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_h_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_s_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])
            
            avg_h_vel_norm = (np.linalg.norm(l_h_vel_matrix, axis=0) + np.linalg.norm(r_h_vel_matrix, axis=0)) / 2.0
            avg_s_vel_norm = (np.linalg.norm(l_s_vel_matrix, axis=0) + np.linalg.norm(r_s_vel_matrix, axis=0)) / 2.0

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 22: {e}")
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

        ax22c.plot(sys_time_indices, sys_entropy_raw, 
                   label=f'Avg Raw (L+R) (Avg: {avg_sys_r:.3f})', color='purple', alpha=0.6, linewidth=1)
        ax22c.plot(sys_time_indices, sys_entropy_smooth, 
                   label=f'Avg Smooth (L+R) (Avg: {avg_sys_s:.3f})', color='blue', linewidth=2)
        ax22c.fill_between(sys_time_indices, sys_entropy_smooth, sys_entropy_raw, color='green', alpha=0.1, label='Redundant Noise Removed')
        ax22c.set_title("Average Entropy (Left + Right Arms)")
        ax22c.set_xlabel("Time Steps")
        ax22c.legend(loc='upper right')
        ax22c.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 23 (UPDATED: 6-DOF Intent - Left, Right, Avg) ---
        print("Generating Figure 23: 6-DOF Action Analysis (Smoothed Intent vs Raw Haptic)")

        fig23, (ax23a, ax23b, ax23c) = plt.subplots(1, 3, figsize=(18, 6))
        fig23.suptitle("Figure 23: 6-DOF Action Analysis (Smoothed Intent vs Raw Haptic)", fontsize=16)

        def compute_6dof_norm(matrix):
            return np.linalg.norm(matrix, axis=0)

        def plot_intent_6dof_subplot(ax, raw_matrix, intent_matrix, title, label_prefix):
            raw_norm = compute_6dof_norm(raw_matrix)
            intent_norm = compute_6dof_norm(intent_matrix)

            avg_raw = np.mean(raw_norm)
            avg_intent = np.mean(intent_norm)
            
            steps_arg = steps 
            
            ax.plot(steps_arg, raw_norm, 
                   label=f"Raw Haptic (Avg: {avg_raw:.3f})", color='gray', linestyle='--', alpha=0.6)
            ax.plot(steps_arg, intent_norm, 
                   label=f"Robot Transformer Model Intent (Avg: {avg_intent:.3f})", linewidth=1.5)
            ax.fill_between(steps_arg, intent_norm, raw_norm, color='skyblue', alpha=0.2, label='Planning Delta')
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("6-DOF Magnitude (L2 Norm)")

        try:
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_smooth_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_smooth_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 23: {e}")
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

        # --- FIGURE 24 (UPDATED: 3 Subplots - Left, Right, Avg) ---
        print("Generating Figure 24: Explainability AI - Adaptive Compliance (System Stiffness)")

        fig24, (ax24a, ax24b, ax24c) = plt.subplots(1, 3, figsize=(18, 6))
        fig24.suptitle("Figure 24: Explainability AI - Adaptive Compliance (6-DOF Speed)", fontsize=16)

        def plot_compliance_subplot(ax, data, title, label_prefix):
            avg_comp = np.mean(data)
            
            ax.plot(steps[:len(data)], data, label=f'{label_prefix} Compliance (Avg: {avg_comp:.3f})', 
                   color='purple', linewidth=2)
            ax.fill_between(steps[:len(data)], data, 0, color='purple', alpha=0.2, 
                         label=f'Interaction Area')
            ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Neutral Threshold')
            ax.axhline(y=0.1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Soft Limit (Safety)')
            ax.axhline(y=0.9, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Stiff Limit (High Speed)')
            ax.set_title(title)
            ax.set_ylabel("Compliance Factor (0=Stiff, 1=Soft)")
            ax.set_xlabel("Time Steps")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])

        try:
            comp_data = np.array(list(self.viz.compliance_factor_hist))
        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 24: {e}")
            comp_data = np.zeros(len(steps))

        plot_compliance_subplot(ax24a, comp_data, "Left Arm (System Stiffness)", "Left")
        plot_compliance_subplot(ax24b, comp_data, "Right Arm (System Stiffness)", "Right")
        plot_compliance_subplot(ax24c, comp_data, "Average System Stiffness (Left + Right)", "Avg")

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 25 (FIXED: Stable Percentage Calculation) ---
        print("Generating Figure 25: Explainability of - Kalman Filter Effect (6-DOF + % Change)")

        # Create figure with 1 row and 3 columns
        fig25, (ax25a, ax25b, ax25c) = plt.subplots(1, 3, figsize=(18, 6))
        fig25.suptitle("Figure 25: Kalman Filter Effect (6-DOF L2 Norm)", fontsize=16)

        # Helper function to plot a single axis
        def plot_kalman_subplot(ax, raw_matrix, corrected_matrix, title, label_prefix):
            # Calculate Overall 6-DOF Magnitude (L2 Norm) across joints
            raw_norm = np.linalg.norm(raw_matrix, axis=0)
            corrected_norm = np.linalg.norm(corrected_matrix, axis=0)
            
            # Calculate Correction Delta (Absolute difference)
            correction_delta = np.abs(corrected_norm - raw_norm)
            
            # --- FIX: Calculate Percentage relative to Average Signal Magnitude ---
            # Using instantaneous raw_norm as denominator causes spikes (infinity) when velocity is 0.
            # We calculate the percentage of the average signal magnitude for a stable metric.
            avg_raw_mag = np.mean(raw_norm)
            epsilon = 1e-9
            
            if avg_raw_mag < epsilon:
                avg_pct_change = 0.0
            else:
                # Calculate percentage relative to the arm's average activity level
                pct_change = (correction_delta / avg_raw_mag) * 100
                avg_pct_change = np.mean(pct_change)

            # Calculate Averages for Legend (Magnitude)
            avg_raw = np.mean(raw_norm)
            avg_corr = np.mean(corrected_norm)
            
            # Plot Raw Input
            ax.plot(steps[:len(raw_norm)], raw_norm, 
                   label=f'{label_prefix} Raw Input (Avg: {avg_raw:.4f})', color='orange', alpha=0.5, linestyle='--')
            
            # Plot Kalman Output
            ax.plot(steps[:len(corrected_norm)], corrected_norm, 
                   label=f'{label_prefix} Kalman Output (Avg: {avg_corr:.4f})', color='blue', linewidth=1.5)
            
            # Fill between to show the Correction Delta
            # Label updated to reflect the stable % calculation
            ax.fill_between(steps[:len(corrected_norm)], corrected_norm, raw_norm, 
                           color='gray', alpha=0.1, 
                           label=f'Correction Delta (Avg Red: {avg_pct_change:.1f}%)')
            
            ax.set_title(title)
            ax.set_ylabel("6-DOF Velocity Magnitude")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)

        try:
            # 1. Extract Left Arm Data (All 6 Joints)
            l_raw_vel_matrix = np.array([list(self.viz.left_h_vel_hist[i]) for i in range(6)])
            l_corr_vel_matrix = np.array([list(self.viz.left_s_vel_hist[i]) for i in range(6)])
            
            # 2. Extract Right Arm Data (All 6 Joints)
            r_raw_vel_matrix = np.array([list(self.viz.right_h_vel_hist[i]) for i in range(6)])
            r_corr_vel_matrix = np.array([list(self.viz.right_s_vel_hist[i]) for i in range(6)])

            # 3. Calculate Average Matrices (Left + Right)
            avg_raw_matrix = (l_raw_vel_matrix + r_raw_vel_matrix) / 2.0
            avg_corr_matrix = (l_corr_vel_matrix + r_corr_vel_matrix) / 2.0

        except (AttributeError, IndexError) as e:
            print(f"Warning: History not found in Viz for Figure 25: {e}")
            l_raw_vel_matrix = np.zeros((6, len(steps)))
            l_corr_vel_matrix = np.zeros((6, len(steps)))
            r_raw_vel_matrix = np.zeros((6, len(steps)))
            r_corr_vel_matrix = np.zeros((6, len(steps)))
            avg_raw_matrix = np.zeros((6, len(steps)))
            avg_corr_matrix = np.zeros((6, len(steps)))

        # 4. Plot Subplots
        # Left Side: Left Arm
        plot_kalman_subplot(ax25a, l_raw_vel_matrix, l_corr_vel_matrix, "Left Arm (6-DOF)", "Left")
        
        # Middle: Right Arm
        plot_kalman_subplot(ax25b, r_raw_vel_matrix, r_corr_vel_matrix, "Right Arm (6-DOF)", "Right")
        
        # Right Side: Average (Left + Right)
        plot_kalman_subplot(ax25c, avg_raw_matrix, avg_corr_matrix, "Average System (Left+Right)", "Avg (L+R)")

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # --- FIGURE 26 (UPDATED: 3 Subplots with % of Safety Limit) ---
        print("Generating Figure 26: Explainability - Safety Limiter (6-DOF L2 Norm)")

        fig26, (ax26a, ax26b, ax26c) = plt.subplots(1, 3, figsize=(18, 6))
        fig26.suptitle("Figure 26: Safety Limiter Analysis (6-DOF L2 Norm)", fontsize=16)

        def plot_safety_subplot(ax, safe_matrix, title, label_prefix):
            safe_norm = np.linalg.norm(safe_matrix, axis=0)
            
            max_limit_per_joint = 0.005
            max_l2_limit = np.sqrt(6) * max_limit_per_joint
            
            avg_safe = np.mean(safe_norm)
            avg_limit_util = (avg_safe / max_l2_limit) * 100
            
            ax.plot(steps[:len(safe_norm)], safe_norm, 
                   label=f'{label_prefix} Clamp Mag (Avg: {avg_safe:.4f})', color='red', alpha=0.6)
            ax.axhline(y=max_l2_limit, color='black', linestyle='--', linewidth=2, label='Max System Limit of Acceleration')
            ax.fill_between(steps[:len(safe_norm)], safe_norm, 0, color='orange', alpha=0.1, 
                           label=f'Limit Utilization (Avg: {avg_limit_util:.1f}%)')
            ax.set_title(title)
            ax.set_ylabel("Safety Clamp Magnitude (L2 Norm)")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)

        try:
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

        plt.tight_layout()
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()      

        pp.close()
        print(f"[SUCCESS] Report saved to {self.pdf_filename}")

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

left_arm_joint_names = [f"fl_joint{i}" for i in range(1, 9)]
left_arm_actuator_ctrl_indices = [model.actuator(name).id for name in left_arm_joint_names]
left_gripper_indices = [model.actuator("fl_joint7").id, model.actuator("fl_joint8").id]

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
        state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01]
    return state, state[6]

print("Syncing state...")
right_arm_state, right_gripper_val = get_and_sync_state(model, data, "fr_joint", right_arm_actuator_ctrl_indices, right_gripper_indices)
left_arm_state, left_gripper_val = get_and_sync_state(model, data, "fl_joint", left_arm_actuator_ctrl_indices, left_gripper_indices)
print("Sync Complete.")

# Gripper Config
GRIPPER_MIN = ctrl_ranges[14][0]
GRIPPER_MAX = ctrl_ranges[14][1]
GRIPPER_STEP = 0.001 

# Initialize IK Solvers
ik_solver_r = CartesianIK(model, "fr_link6")
ik_solver_l = CartesianIK(model, "fl_link6")

# Initialize RL Policy and Visualization
rl_policy = BiActionChunkingPolicy(num_joints=6, chunk_size=20, smoothing_factor=0.2)
visualizer = LearningVisualizer(history_len=200)
reporter = LearningReportGenerator(visualizer, rl_policy)

# =============================================================================
# 4. MAIN LOOP
# =============================================================================

if __name__ == "__main__":
    print("\n=== Haptic Dual Arm Control + RL Analysis ===")
    
    # Scaling Config
    SCALE_J1 = 4.0
    SCALE_J2 = 4.0
    SCALE_J3 = 4.0
    SCALE_J4 = 2.0
    SCALE_J5 = 2.0
    SCALE_J6 = 2.0
    #SCALE_J7 = 1.0
    #SCALE_J8 = 1.0
    
    OFFSET_J2 = 0.0
    OFFSET_J3 = 0.0
    
    print(f"J1 to J3 Scale: 4.0x | Offset: 0.0")
    print(f"J4 to J6 Scale: 2.0x | Offset: 0.0")
    print("Modes: R = Right Arm, L = Left Arm")
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
                # 1. Check Keyboard Input
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

                # 3. Calculate Haptic Velocity (Raw Input for RL Policy)
                raw_delta = h_joints - prev_haptic_joints
                
                # Swap J4/J5 Deltas to match Robot Orientation Fix
                temp_delta_3 = raw_delta[3]
                raw_delta[3] = raw_delta[4] 
                raw_delta[4] = temp_delta_3 

                # 4. RL Policy Smoothing
                refined_delta, intent, kalman, comp, safe = rl_policy.predict(raw_delta)

                # 5. Integration (Apply Velocity to Position)
                if active_arm == "right":
                    right_arm_state[0] += refined_delta[0] * SCALE_J1
                    right_arm_state[1] += refined_delta[1] * SCALE_J2 
                    right_arm_state[2] += refined_delta[2] * SCALE_J3 
                    right_arm_state[3] += refined_delta[3] * SCALE_J4
                    right_arm_state[4] += refined_delta[4] * SCALE_J5
                    right_arm_state[5] += refined_delta[5] * SCALE_J6

                    if h_btn_top: right_gripper_val -= GRIPPER_STEP
                    elif h_btn_bottom: right_gripper_val += GRIPPER_STEP
                    right_gripper_val = clamp(right_gripper_val, GRIPPER_MIN, GRIPPER_MAX)
                    right_arm_state[6] = right_gripper_val
                    right_arm_state[7] = right_gripper_val

                    for i in range(6):
                        data.ctrl[right_arm_actuator_ctrl_indices[i]] = clamp(right_arm_state[i], ctrl_ranges[i][0], ctrl_ranges[i][1])
                    data.ctrl[right_gripper_indices[0]] = right_arm_state[6]
                    data.ctrl[right_gripper_indices[1]] = right_arm_state[7]

                    # VISUALIZER UPDATE (FULL DATA)
                    visualizer.update(h_joints, right_arm_state, raw_delta, refined_delta, intent, kalman, comp, safe, right_arm_state[6], right_arm_state[7], arm_name="right")
                    
                    render_cam = cam_right_id
                    cam_text = "RIGHT (RL Active)"

                elif active_arm == "left":
                    left_arm_state[0] += refined_delta[0] * SCALE_J1
                    left_arm_state[1] += refined_delta[1] * SCALE_J2
                    left_arm_state[2] += refined_delta[2] * SCALE_J3
                    left_arm_state[3] += refined_delta[3] * SCALE_J4
                    left_arm_state[4] += refined_delta[4] * SCALE_J5
                    left_arm_state[5] += refined_delta[5] * SCALE_J6

                    if h_btn_top: left_gripper_val -= GRIPPER_STEP
                    elif h_btn_bottom: left_gripper_val += GRIPPER_STEP
                    left_gripper_val = clamp(left_gripper_val, GRIPPER_MIN, GRIPPER_MAX)
                    left_arm_state[6] = left_gripper_val
                    left_arm_state[7] = left_gripper_val

                    for i in range(6):
                        data.ctrl[left_arm_actuator_ctrl_indices[i]] = clamp(left_arm_state[i], ctrl_ranges[i+8][0], ctrl_ranges[i+8][1])
                    data.ctrl[left_gripper_indices[0]] = left_arm_state[6]
                    data.ctrl[left_gripper_indices[1]] = left_arm_state[7]

                    visualizer.update(h_joints, left_arm_state, raw_delta, refined_delta, intent, kalman, comp, safe, left_arm_state[6], left_arm_state[7], arm_name="left")

                    render_cam = cam_left_id
                    cam_text = "LEFT (RL Active)"

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

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if 'haptic_device' in locals():
            haptic_device.close()
        
        print("\nSimulation finished. Generating Learning Report...")
        print("NOTE: 26 Pop-up windows will appear. Close them manually.")
        reporter.generate_report()
        print("Done.")
