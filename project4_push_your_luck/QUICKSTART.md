# 🚀 Quick Start Guide - Push Your Luck!

## Get Started in 3 Minutes

### Step 1: Install Dependencies (30 seconds)
```bash
cd project4_push_your_luck
pip install -r requirements.txt
```

### Step 2: View Results (Instant)
Results are already generated! Just visualize them:

```bash
# Option A: Static plots
python visualize_results.py

# Option B: Interactive dashboard
streamlit run app.py
```

### Step 3: Explore (Optional)

#### Test the Environment
```bash
python environment.py
```

#### Train Your Own Agents
```bash
# Quick training (500 episodes, ~10 seconds)
python quick_train.py

# Full training (1000 episodes, ~20 seconds)
python training.py
```

---

## 📊 What You'll See

### Training Results Summary
```
🏆 Best Performer: Actor-Critic
   Average Return: 16.84
   Average Treasure: 10.12

🥈 Runner-up: Q-Learning
   Average Return: 15.00
   Training Time: 0.05s (fastest!)
```

### Generated Visualizations
- **training_curves.png**: 4-panel training progress
- **performance_comparison.png**: Agent comparison bar charts
- **baseline_effect.png**: REINFORCE baseline analysis

### Interactive Dashboard Features
1. **Training Dashboard**: View all metrics and learning curves
2. **Interactive Demo**: Play the game yourself!
3. **Analysis**: Deep dive into performance
4. **About**: Complete documentation

---

## 🎮 How to Play (Interactive Demo)

1. Launch dashboard: `streamlit run app.py`
2. Navigate to "🎮 Interactive Demo"
3. Click "🆕 New Game"
4. Choose your action:
   - **🛑 STOP**: Keep your treasure (safe)
   - **🎲 ROLL**: Roll dice again (risky!)
5. Watch the dice outcomes and treasure changes

### Dice Outcomes
- **1-3**: 🏆 Treasure (+10 per die)
- **4**: ⚠️ Minor risk (-10% treasure)
- **5**: 🔥 Major risk (-30% treasure)
- **6**: 💀 Game over (lose all!)

---

## 📁 Project Structure

```
project4_push_your_luck/
├── environment.py          # Game environment
├── agents.py              # RL algorithms
├── training.py            # Training script
├── app.py                 # Streamlit dashboard
├── visualize_results.py   # Plot generator
├── quick_train.py         # Fast training
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── PROJECT_SUMMARY.md    # Complete summary
├── QUICKSTART.md         # This file
└── results/              # Training results
    ├── actor_critic.json
    ├── qlearning.json
    ├── reinforce_baseline.json
    └── reinforce_no_baseline.json
```

---

## 🎯 Key Results at a Glance

| Agent | Performance | Speed | Best For |
|-------|------------|-------|----------|
| **Actor-Critic** | 🥇 Best (16.84) | Medium | Overall performance |
| **Q-Learning** | 🥈 Good (15.00) | ⚡ Fastest | Quick training |
| REINFORCE | ⚠️ Conservative | Slow | Learning example |

---

## 💡 Quick Tips

### Want Better REINFORCE Results?
Edit `agents.py` and try:
- Increase learning rate: `learning_rate=0.01`
- Add entropy bonus for exploration
- Train longer: `num_episodes=2000`

### Want to Modify the Game?
Edit `environment.py`:
```python
env = DiceExplorationEnv(
    num_dice=5,              # More dice!
    max_turns=20,            # Longer episodes
    enable_learning_variant=True  # Experience reduces risk
)
```

### Want Different Visualizations?
Edit `visualize_results.py` and customize:
- Colors, line styles, plot layouts
- Moving average windows
- Additional metrics

---

## ❓ Troubleshooting

### "No module named 'streamlit'"
```bash
pip install streamlit plotly
```

### "No training results found"
```bash
python quick_train.py
```

### Dashboard won't open
Check if port 8501 is available, or specify another:
```bash
streamlit run app.py --server.port 8502
```

### Plots not showing
Make sure matplotlib is installed:
```bash
pip install matplotlib
```

---

## 🎓 What You're Learning

This project demonstrates:
- ✅ **Policy Gradient Methods** (REINFORCE, Actor-Critic)
- ✅ **Value-Based Methods** (Q-Learning)
- ✅ **Baseline Techniques** (Variance reduction)
- ✅ **MDP Design** (States, actions, rewards)
- ✅ **Experimental Analysis** (Comparing algorithms)

---

## 🚀 Next Steps

1. **Explore the Dashboard**: See all visualizations
2. **Play the Game**: Understand the environment
3. **Read the Code**: Learn implementation details
4. **Experiment**: Modify parameters and retrain
5. **Present**: Use visualizations for your presentation

---

## 📞 Need Help?

Check these files:
- **README.md**: Comprehensive documentation
- **PROJECT_SUMMARY.md**: Complete analysis
- **Code comments**: Inline explanations

---

**Ready? Let's go!** 🎲

```bash
streamlit run app.py
```

**Made with ❤️ for Reinforcement Learning**
