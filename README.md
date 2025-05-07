# Availability-Aware-Network-Slicing-and-Dynamic-Function

**SliAvailRAN** (Availability-Aware Slicing and Adaptive Function Placement in Virtualized RANs using Reinforcement Learning) is a reinforcement learning-based framework that optimizes network slicing and function placement in virtualized Radio Access Networks (vRANs). This project addresses challenges in 5G networks such as high reliability, scalability, and efficient resource utilization using Proximal Policy Optimization (PPO).

## 🧠 Features

- 📡 Availability-Aware Network Slicing with disjoint primary/backup paths
- 🔄 Dynamic CU/DU function placement based on live resource availability
- 🧠 Reinforcement Learning with PPO (via Stable-Baselines3)
- ⚠️ Constraint-aware decision making (latency, bandwidth, wavelength continuity)
- 📈 Scalable to large topologies (up to 128 nodes)

## 📂 Project Structure

sliavailran/<br>
├── rl_agent/ # PPO-based RL agent logic<br>
├── environment/ # Custom Gym-compatible simulation<br>
├── config/ # Network topology and RL hyperparameters<br>
├── scripts/ # Training and evaluation scripts<br>
├── models/ # Saved RL models<br>
├── results/ # Output results, plots, metrics<br>
├── utils/ # Helper functions (rewards, constraints, state mgmt)<br>
├── *.json # Network topology files<br>
└── README.md # This file<br>

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sliavailran.git
cd sliavailran
```
### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install gymnasium
pip install numpy
pip install pandas
pip install matplotlib
pip install stable-baselines3
```
### 4. Evaluate the Model
```bash
python evaluate_ppo.py
```
## 📊 Evaluation Metrics

- ✅ Slice Acceptance Rate
- 📉 Service Drop Rate
- 🧠 Reward Convergence
- ⚙️ VNC Allocation Statistics
- ⏱ Training and Inference Time
