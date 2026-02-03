ALOHA Haptic Teleoperation with Adaptive RL Policy
This project implements a high-fidelity dual-arm teleoperation system for the ALOHA robot, controlled via 3D Systems Haptic devices. It features a custom Bi-Action Chunking Policy that combines Temporal Transformers, Graph Attention Networks (GAT), Meta-Learning, and Kalman Filtering to ensure smooth, compliant, and stable motion.


Features:
Haptic Teleoperation: Direct control of ALOHA dual arms using 3D Systems Geomagic/Touch devices.

Bi-Action Chunking Policy: A complex RL-style architecture that processes action history to generate smooth trajectories.

Graph Attention Network (GAT): Captures spatial dependencies between joints for coordinated motion.

Meta-Learning Adaptation: Dynamically adjusts trust between the raw haptic input and the learned policy based on real-time error.

Kalman Filtering: Provides optimal estimation and smoothing of position and velocity signals.

Adaptive Compliance: Modulates stiffness based on velocity; the robot becomes "softer" at low speeds for safe interaction.

Automated Reporting: Generates a comprehensive PDF report analyzing motion smoothness, frequency attenuation, jerk, and control effort after each session.

Meshes & Assets Setup

The code relies on the MuJoCo XML and mesh files for the ALOHA robot. Since the robot models are large binary assets, they are not included in this repository.

Please follow these steps to get the meshes:

Download the Repository:Go to the official mobile_aloha_sim repository and download it as a ZIP, or clone it:

git clone https://github.com/agilexrobotics/mobile_aloha_sim.git

Locate the Meshes:Navigate to the following folder inside the downloaded repository:

mobile_aloha_sim-master/aloha_mujoco/aloha/meshes_mujoco/

You specifically need the folder containing aloha_v1.xml and the associated meshes subdirectory.

Update Code Path:Open the python script and find the XML_PATH variable (around line 430). Update it to point to your downloaded file:

XML_PATH = "/path/to/your/download/mobile_aloha_sim-master/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"

🛠️ Installation
Prerequisites

OS: Ubuntu (Tested on 20.04/22.04)

Python: 3.8+

Hardware: 3D Systems Geomagic Touch / Omni Haptic Device

Drivers: 3D Systems OpenHaptics Toolkit (must be installed and libHD.so available)

Python Dependencies

Install the required Python packages:

pip install numpy opencv-python matplotlib scipy pyOpenHaptics

Ensure mujoco is installed (preferably via the official source or pip install mujoco for the latest version).

Usage

Connect Device: Ensure your haptic device is connected and powered on.

Run the Script:

bash
Main.py

Controls:

r: Switch control to Right Arm.

l: Switch control to Left Arm.

ESC: Exit simulation and generate report.

Haptic Buttons: Top/Bottom buttons control the gripper open/close.

View Results: Upon exit, several pop-up windows will appear displaying graphs. Close them to finalize the PDF report generation on your Desktop.

License & Credit
Robot Model: AgileXRobotics / mobile_aloha_sim
Haptics: 3D Systems OpenHaptics
