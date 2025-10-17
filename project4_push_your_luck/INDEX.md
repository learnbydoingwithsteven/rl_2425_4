# 🎲 Push Your Luck! - Project Index

## 📋 Complete File Listing

### 📖 Documentation
| File | Description | Size |
|------|-------------|------|
| **QUICKSTART.md** | 🚀 Get started in 3 minutes | Essential |
| **README.md** | 📚 Comprehensive project guide | Full docs |
| **PROJECT_SUMMARY.md** | 📊 Complete analysis & results | Detailed |
| **INDEX.md** | 📋 This file | Navigation |

### 💻 Core Implementation
| File | Description | Lines | Purpose |
|------|-------------|-------|---------|
| **environment.py** | 🎮 Game environment (MDP) | ~350 | Dice game logic |
| **agents.py** | 🤖 RL algorithms | ~450 | REINFORCE, Actor-Critic, Q-Learning |
| **training.py** | 🏋️ Training infrastructure | ~300 | Train all agents |
| **app.py** | 📱 Interactive dashboard | ~650 | Streamlit UI |

### 🛠️ Utilities
| File | Description | Purpose |
|------|-------------|---------|
| **visualize_results.py** | 📊 Plot generator | Create static visualizations |
| **quick_train.py** | ⚡ Fast training | 500 episodes (~10s) |
| **requirements.txt** | 📦 Dependencies | pip install |

### 📊 Results & Visualizations
| File | Type | Description |
|------|------|-------------|
| **training_curves.png** | 📈 Plot | 4-panel training progress |
| **performance_comparison.png** | 📊 Plot | Agent comparison bars |
| **baseline_effect.png** | 📉 Plot | REINFORCE baseline analysis |
| **results/actor_critic.json** | 💾 Data | Actor-Critic training history |
| **results/qlearning.json** | 💾 Data | Q-Learning training history |
| **results/reinforce_baseline.json** | 💾 Data | REINFORCE (baseline) history |
| **results/reinforce_no_baseline.json** | 💾 Data | REINFORCE (no baseline) history |

---

## 🎯 Quick Navigation

### I want to...

#### 🚀 **Get started quickly**
→ Read **QUICKSTART.md**

#### 📚 **Understand the project**
→ Read **README.md**

#### 📊 **See the results**
→ Read **PROJECT_SUMMARY.md**  
→ View **training_curves.png**, **performance_comparison.png**

#### 🎮 **Play the game**
→ Run `streamlit run app.py`  
→ Navigate to "Interactive Demo"

#### 🔬 **Understand the code**
→ Read **environment.py** (game logic)  
→ Read **agents.py** (algorithms)  
→ Read **training.py** (training loop)

#### 📈 **Generate new plots**
→ Run `python visualize_results.py`

#### 🏋️ **Train new agents**
→ Run `python quick_train.py` (fast)  
→ Run `python training.py` (full)

#### 🎨 **Customize visualizations**
→ Edit **visualize_results.py**  
→ Edit **app.py** (dashboard)

---

## 📊 Project Statistics

### Code Metrics
- **Total Python Files**: 6
- **Total Lines of Code**: ~2,200
- **Documentation Files**: 4
- **Visualization Files**: 3
- **Training Results**: 4 JSON files

### Training Results
- **Agents Trained**: 4
- **Total Episodes**: 2,000 (500 each)
- **Training Time**: ~9 seconds
- **Best Agent**: Actor-Critic (16.84 avg return)
- **Fastest Agent**: Q-Learning (0.05s training)

### Visualizations
- **Static Plots**: 3 PNG files (1.6 MB total)
- **Interactive Dashboard**: Full Streamlit app
- **Plot Resolution**: 300 DPI (publication quality)

---

## 🎓 Learning Path

### Beginner
1. Read **QUICKSTART.md**
2. Run `python environment.py` (test game)
3. Run `streamlit run app.py` (explore dashboard)
4. Play in "Interactive Demo"

### Intermediate
1. Read **README.md** (full documentation)
2. Study **environment.py** (MDP design)
3. Study **agents.py** (algorithm implementations)
4. Run `python quick_train.py` (train agents)
5. View **PROJECT_SUMMARY.md** (analysis)

### Advanced
1. Read all code files with comments
2. Modify hyperparameters in **agents.py**
3. Customize environment in **environment.py**
4. Implement new algorithms
5. Conduct experiments and analyze results

---

## 🔍 File Details

### environment.py
**Purpose**: Implements the Push Your Luck! game as an MDP

**Key Classes**:
- `DiceExplorationEnv`: Main environment class

**Key Methods**:
- `reset()`: Start new episode
- `step(action)`: Take action, get reward
- `_roll_dice()`: Simulate dice rolls
- `get_state_vector()`: Convert state to neural network input

**Features**:
- Configurable dice count and max turns
- Learning variant with experience-based risk reduction
- Comprehensive state representation
- Detailed step information

### agents.py
**Purpose**: Implements RL algorithms

**Key Classes**:
- `REINFORCEAgent`: Monte Carlo policy gradient
- `ActorCriticAgent`: Policy gradient with value function
- `QLearningAgent`: Value-based tabular method
- `PolicyNetwork`: Neural network for policy
- `ValueNetwork`: Neural network for value function

**Key Methods**:
- `select_action()`: Choose action from policy
- `update()`: Update agent parameters
- `store_reward()`: Store reward (REINFORCE)

### training.py
**Purpose**: Training infrastructure for all agents

**Key Classes**:
- `Trainer`: Unified training interface

**Key Methods**:
- `train_reinforce()`: Train REINFORCE agent
- `train_actor_critic()`: Train Actor-Critic agent
- `train_qlearning()`: Train Q-Learning agent
- `save_results()`: Save training history to JSON

### app.py
**Purpose**: Interactive Streamlit dashboard

**Key Functions**:
- `show_training_dashboard()`: Display training results
- `show_interactive_demo()`: Manual gameplay
- `show_analysis()`: Detailed performance analysis
- `show_about()`: Project documentation
- `plot_training_curves()`: Generate Plotly charts
- `plot_performance_comparison()`: Bar chart comparisons

### visualize_results.py
**Purpose**: Generate static matplotlib visualizations

**Key Functions**:
- `load_results()`: Load JSON training data
- `plot_training_curves()`: 4-panel training progress
- `plot_performance_comparison()`: 3-panel bar charts
- `plot_baseline_effect()`: REINFORCE comparison
- `print_summary()`: Console output with statistics

---

## 🎨 Visualization Guide

### Training Curves (training_curves.png)
**4 Panels**:
1. **Top-Left**: Raw episode returns (all agents)
2. **Top-Right**: Moving average returns (window=50)
3. **Bottom-Left**: Episode lengths over time
4. **Bottom-Right**: Final treasures over time

**Colors**:
- 🔴 REINFORCE (no baseline): #FF6B6B
- 🔵 REINFORCE (baseline): #4ECDC4
- 🟢 Actor-Critic: #45B7D1
- 🟠 Q-Learning: #FFA07A

### Performance Comparison (performance_comparison.png)
**3 Bar Charts**:
1. **Left**: Average return (last 100 episodes)
2. **Center**: Average treasure (last 100 episodes)
3. **Right**: Training time (seconds)

### Baseline Effect (baseline_effect.png)
**Line Plot**:
- Compares REINFORCE with/without baseline
- Shows moving average (window=50)
- Demonstrates variance reduction effect

---

## 📦 Dependencies

```
numpy>=1.24.0       # Numerical computing
torch>=2.0.0        # Neural networks
matplotlib>=3.7.0   # Static plots
plotly>=5.14.0      # Interactive plots
streamlit>=1.28.0   # Dashboard
pandas>=2.0.0       # Data manipulation
tqdm>=4.65.0        # Progress bars
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Examples

### Example 1: Quick Start
```bash
# Install and run
pip install -r requirements.txt
streamlit run app.py
```

### Example 2: Train and Visualize
```bash
# Train agents
python quick_train.py

# Generate plots
python visualize_results.py

# View dashboard
streamlit run app.py
```

### Example 3: Custom Training
```python
from environment import DiceExplorationEnv
from agents import ActorCriticAgent

# Create environment
env = DiceExplorationEnv(num_dice=5, max_turns=15)

# Create agent
agent = ActorCriticAgent(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    actor_lr=0.001,
    critic_lr=0.005
)

# Train
for episode in range(1000):
    state_dict = env.reset()
    # ... training loop
```

### Example 4: Manual Gameplay
```python
from environment import DiceExplorationEnv

env = DiceExplorationEnv()
state = env.reset()

while not state["done"]:
    action = int(input("Action (0=STOP, 1=ROLL): "))
    state, reward, done, info = env.step(action)
    print(f"Reward: {reward}, {info['message']}")
```

---

## 📞 Support

### Documentation
- **QUICKSTART.md**: Fast setup
- **README.md**: Complete guide
- **PROJECT_SUMMARY.md**: Detailed analysis
- **Code comments**: Inline explanations

### Common Issues
1. **Import errors**: Run `pip install -r requirements.txt`
2. **No results**: Run `python quick_train.py`
3. **Dashboard errors**: Check Streamlit version
4. **Plot errors**: Check matplotlib installation

---

## ✅ Checklist

### For Presentation
- [ ] Read PROJECT_SUMMARY.md
- [ ] View all 3 visualization PNGs
- [ ] Run interactive dashboard
- [ ] Prepare slides with key results
- [ ] Practice explaining algorithms

### For Submission
- [ ] All code files present
- [ ] Training results generated
- [ ] Visualizations created
- [ ] Documentation complete
- [ ] GitHub repository ready

### For Understanding
- [ ] Understand MDP formulation
- [ ] Understand policy gradient methods
- [ ] Understand Actor-Critic architecture
- [ ] Understand baseline techniques
- [ ] Understand experimental results

---

## 🎯 Key Takeaways

1. **Actor-Critic** performs best (16.84 avg return)
2. **Q-Learning** is fastest (0.05s training)
3. **REINFORCE** needs tuning (learned conservative policy)
4. **Baseline** helps but not sufficient alone
5. **Policy gradients** work well for this problem

---

**Project Complete! 🎉**

Ready for presentation and submission.

---

**Made with ❤️ for Reinforcement Learning**  
**Università degli Studi di Milano - 2024-25**
