"""
Push Your Luck! - Interactive Streamlit Dashboard
Complete visualization and interaction for Project 4
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List
import time

from environment import DiceExplorationEnv
from agents import REINFORCEAgent, ActorCriticAgent, QLearningAgent
from training import Trainer


# Page configuration
st.set_page_config(
    page_title="Push Your Luck! - RL Project 4",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


def load_training_results() -> Dict[str, Dict]:
    """Load all training results from JSON files."""
    results = {}
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        return results
    
    files = {
        "REINFORCE (no baseline)": "reinforce_no_baseline.json",
        "REINFORCE (baseline)": "reinforce_baseline.json",
        "Actor-Critic": "actor_critic.json",
        "Q-Learning": "qlearning.json"
    }
    
    for name, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[name] = json.load(f)
    
    return results


def plot_training_curves(results: Dict[str, Dict]):
    """Plot training curves for all agents."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Episode Returns", "Moving Average Returns (window=50)",
                       "Episode Lengths", "Final Treasures"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = {
        "REINFORCE (no baseline)": "#FF6B6B",
        "REINFORCE (baseline)": "#4ECDC4",
        "Actor-Critic": "#45B7D1",
        "Q-Learning": "#FFA07A"
    }
    
    for agent_name, history in results.items():
        color = colors.get(agent_name, "#888888")
        episodes = list(range(len(history["returns"])))
        
        # Raw returns
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=history["returns"],
                name=agent_name,
                line=dict(color=color, width=1),
                opacity=0.3,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Moving average
        window = 50
        if len(history["returns"]) >= window:
            ma_returns = pd.Series(history["returns"]).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=ma_returns,
                    name=f"{agent_name} (MA)",
                    line=dict(color=color, width=3),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Episode lengths
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=history["episode_lengths"],
                name=agent_name,
                line=dict(color=color, width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Treasures
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=history["treasures"],
                name=agent_name,
                line=dict(color=color, width=1),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Return (MA)", row=1, col=2)
    fig.update_yaxes(title_text="Steps", row=2, col=1)
    fig.update_yaxes(title_text="Treasure", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="Training Progress Comparison",
        title_font_size=20,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def plot_performance_comparison(results: Dict[str, Dict]):
    """Plot performance comparison across agents."""
    agent_names = list(results.keys())
    
    # Calculate metrics
    final_returns = [np.mean(results[name]["returns"][-100:]) for name in agent_names]
    final_treasures = [np.mean(results[name]["treasures"][-100:]) for name in agent_names]
    training_times = [results[name]["training_time"] for name in agent_names]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Average Return (last 100)", "Average Treasure (last 100)", "Training Time (s)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    
    # Returns
    fig.add_trace(
        go.Bar(x=agent_names, y=final_returns, marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # Treasures
    fig.add_trace(
        go.Bar(x=agent_names, y=final_treasures, marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # Training time
    fig.add_trace(
        go.Bar(x=agent_names, y=training_times, marker_color=colors, showlegend=False),
        row=1, col=3
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, title_text="Performance Comparison", title_font_size=20)
    
    return fig


def plot_baseline_effect(results: Dict[str, Dict]):
    """Plot effect of baseline on REINFORCE."""
    if "REINFORCE (no baseline)" not in results or "REINFORCE (baseline)" not in results:
        return None
    
    fig = go.Figure()
    
    # No baseline
    no_baseline = results["REINFORCE (no baseline)"]
    episodes = list(range(len(no_baseline["returns"])))
    ma_no_baseline = pd.Series(no_baseline["returns"]).rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=episodes,
        y=ma_no_baseline,
        name="Without Baseline",
        line=dict(color="#FF6B6B", width=3)
    ))
    
    # With baseline
    baseline = results["REINFORCE (baseline)"]
    ma_baseline = pd.Series(baseline["returns"]).rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=episodes,
        y=ma_baseline,
        name="With Baseline",
        line=dict(color="#4ECDC4", width=3)
    ))
    
    fig.update_layout(
        title="Effect of Baseline on REINFORCE Performance",
        xaxis_title="Episode",
        yaxis_title="Moving Average Return (window=50)",
        height=400,
        showlegend=True
    )
    
    return fig


def simulate_episode(env: DiceExplorationEnv, agent, agent_type: str) -> List[Dict]:
    """Simulate one episode and return step-by-step information."""
    state_dict = env.reset()
    state = env.get_state_vector(state_dict)
    
    episode_data = []
    step = 0
    
    while not state_dict["done"]:
        # Select action
        if agent_type == "Q-Learning":
            action = agent.select_action(state, training=False)
        else:
            action = agent.select_action(state, training=False)
            if isinstance(action, tuple):
                action = action[0]
        
        # Take step
        state_dict, reward, done, info = env.step(action)
        next_state = env.get_state_vector(state_dict)
        
        # Store step data
        step_data = {
            "step": step,
            "action": "STOP" if action == 0 else "ROLL",
            "reward": reward,
            "treasure": state_dict["treasure"],
            "turn": state_dict["turn"],
            "done": done,
            "info": info.get("message", "")
        }
        
        if "dice_results" in info:
            step_data["dice"] = info["dice_results"]
        
        episode_data.append(step_data)
        
        state = next_state
        step += 1
        
        if done:
            break
    
    return episode_data


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🎲 Push Your Luck! 🎲</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Space Exploration Dice Game - Policy Gradient Methods</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Training Dashboard", "🎮 Interactive Demo", "📈 Analysis", "ℹ️ About"]
    )
    
    if page == "📊 Training Dashboard":
        show_training_dashboard()
    elif page == "🎮 Interactive Demo":
        show_interactive_demo()
    elif page == "📈 Analysis":
        show_analysis()
    else:
        show_about()


def show_training_dashboard():
    """Show training dashboard with results."""
    st.markdown('<div class="sub-header">Training Dashboard</div>', unsafe_allow_html=True)
    
    # Check if results exist
    results = load_training_results()
    
    if not results:
        st.warning("⚠️ No training results found. Please run training first.")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training all agents... This may take a few minutes."):
                # Create progress bars
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize environment and trainer
                env = DiceExplorationEnv(num_dice=3, max_turns=10, enable_learning_variant=False)
                trainer = Trainer(env, save_dir="results")
                
                num_episodes = 1000
                agents_to_train = [
                    ("REINFORCE (no baseline)", lambda: trainer.train_reinforce(num_episodes, False)),
                    ("REINFORCE (baseline)", lambda: trainer.train_reinforce(num_episodes, True)),
                    ("Actor-Critic", lambda: trainer.train_actor_critic(num_episodes)),
                    ("Q-Learning", lambda: trainer.train_qlearning(num_episodes))
                ]
                
                for i, (name, train_func) in enumerate(agents_to_train):
                    status_text.text(f"Training {name}...")
                    history = train_func()
                    trainer.save_results(history, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json")
                    progress_bar.progress((i + 1) / len(agents_to_train))
                
                status_text.text("Training completed! ✅")
                st.success("All agents trained successfully!")
                st.rerun()
        
        return
    
    # Display results
    st.success(f"✅ Loaded results for {len(results)} agents")
    
    # Summary metrics
    st.markdown("### 📊 Performance Summary")
    cols = st.columns(len(results))
    
    for i, (agent_name, history) in enumerate(results.items()):
        with cols[i]:
            avg_return = np.mean(history["returns"][-100:])
            avg_treasure = np.mean(history["treasures"][-100:])
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{agent_name}</h3>
                <p style="font-size: 2rem; margin: 0;">{avg_return:.1f}</p>
                <p style="margin: 0;">Avg Return</p>
                <p style="font-size: 1.5rem; margin-top: 1rem;">{avg_treasure:.1f}</p>
                <p style="margin: 0;">Avg Treasure</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Training curves
    st.markdown("### 📈 Training Curves")
    fig_training = plot_training_curves(results)
    st.plotly_chart(fig_training, use_container_width=True)
    
    # Performance comparison
    st.markdown("### 🏆 Performance Comparison")
    fig_comparison = plot_performance_comparison(results)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Baseline effect
    st.markdown("### 🎯 Effect of Baseline (REINFORCE)")
    fig_baseline = plot_baseline_effect(results)
    if fig_baseline:
        st.plotly_chart(fig_baseline, use_container_width=True)
        
        with st.expander("📝 Analysis"):
            st.write("""
            **Baseline in REINFORCE:**
            - Reduces variance in policy gradient estimates
            - Helps stabilize training and improve convergence
            - Typically implemented as running average of returns
            - Compare the smoothness and final performance of both curves
            """)


def show_interactive_demo():
    """Show interactive demo of trained agents."""
    st.markdown('<div class="sub-header">Interactive Demo</div>', unsafe_allow_html=True)
    
    # Environment settings
    col1, col2, col3 = st.columns(3)
    with col1:
        num_dice = st.slider("Number of Dice", 2, 5, 3)
    with col2:
        max_turns = st.slider("Max Turns", 5, 20, 10)
    with col3:
        learning_variant = st.checkbox("Enable Learning Variant", value=False)
    
    # Create environment
    env = DiceExplorationEnv(
        num_dice=num_dice,
        max_turns=max_turns,
        enable_learning_variant=learning_variant
    )
    
    st.info(f"🎲 Environment: {num_dice} dice, {max_turns} max turns, Learning variant: {learning_variant}")
    
    # Manual play mode
    st.markdown("### 🎮 Manual Play")
    
    if "game_state" not in st.session_state:
        st.session_state.game_state = None
        st.session_state.game_history = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆕 New Game", type="primary"):
            st.session_state.game_state = env.reset()
            st.session_state.game_history = []
            st.rerun()
    
    if st.session_state.game_state is not None:
        state = st.session_state.game_state
        
        # Display current state
        st.markdown("#### Current State")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("💎 Treasure", state["treasure"])
        with metric_cols[1]:
            st.metric("🔄 Turn", f"{state['turn']}/{state['max_turns']}")
        with metric_cols[2]:
            st.metric("Status", "Done" if state["done"] else "Active")
        with metric_cols[3]:
            if learning_variant and state["experience"]:
                total_exp = sum(state["experience"].values())
                st.metric("📚 Experience", total_exp)
        
        # Actions
        if not state["done"]:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("🛑 STOP (Keep Treasure)", use_container_width=True):
                    state, reward, done, info = env.step(0)
                    st.session_state.game_state = state
                    st.session_state.game_history.append({
                        "action": "STOP",
                        "reward": reward,
                        "info": info
                    })
                    st.rerun()
            
            with action_cols[1]:
                if st.button("🎲 ROLL (Push Luck)", use_container_width=True):
                    state, reward, done, info = env.step(1)
                    st.session_state.game_state = state
                    st.session_state.game_history.append({
                        "action": "ROLL",
                        "reward": reward,
                        "info": info
                    })
                    st.rerun()
        
        # Game history
        if st.session_state.game_history:
            st.markdown("#### 📜 Game History")
            for i, step in enumerate(st.session_state.game_history):
                with st.expander(f"Step {i+1}: {step['action']} (Reward: {step['reward']:.1f})"):
                    st.write(step['info']['message'])
                    if 'dice_results' in step['info']:
                        st.write(f"🎲 Dice: {step['info']['dice_results']}")
                    if 'outcomes' in step['info']:
                        st.write(f"Outcomes: {step['info']['outcomes']}")


def show_analysis():
    """Show detailed analysis."""
    st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)
    
    results = load_training_results()
    
    if not results:
        st.warning("⚠️ No training results found. Please run training first.")
        return
    
    # Agent selection
    agent_name = st.selectbox("Select Agent", list(results.keys()))
    history = results[agent_name]
    
    # Statistics
    st.markdown("### 📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes", len(history["returns"]))
    with col2:
        st.metric("Avg Return (last 100)", f"{np.mean(history['returns'][-100:]):.2f}")
    with col3:
        st.metric("Max Return", f"{np.max(history['returns']):.2f}")
    with col4:
        st.metric("Training Time", f"{history['training_time']:.1f}s")
    
    # Distribution plots
    st.markdown("### 📈 Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_returns = px.histogram(
            history["returns"][-500:],
            title="Return Distribution (last 500 episodes)",
            labels={"value": "Return", "count": "Frequency"},
            nbins=50
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        fig_lengths = px.histogram(
            history["episode_lengths"][-500:],
            title="Episode Length Distribution (last 500 episodes)",
            labels={"value": "Length", "count": "Frequency"},
            nbins=20
        )
        st.plotly_chart(fig_lengths, use_container_width=True)
    
    # Learning progress
    st.markdown("### 📉 Learning Progress")
    
    window_size = st.slider("Moving Average Window", 10, 200, 50)
    
    ma_returns = pd.Series(history["returns"]).rolling(window=window_size).mean()
    
    fig_progress = go.Figure()
    fig_progress.add_trace(go.Scatter(
        y=history["returns"],
        name="Raw Returns",
        line=dict(color="lightblue", width=1),
        opacity=0.3
    ))
    fig_progress.add_trace(go.Scatter(
        y=ma_returns,
        name=f"MA (window={window_size})",
        line=dict(color="blue", width=3)
    ))
    fig_progress.update_layout(
        title="Learning Curve",
        xaxis_title="Episode",
        yaxis_title="Return",
        height=400
    )
    st.plotly_chart(fig_progress, use_container_width=True)


def show_about():
    """Show project information."""
    st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎲 Push Your Luck! - Space Exploration Dice Game
    
    ### Project Overview
    This is **Project 4** from the Reinforcement Learning course (2024-25), focusing on **Policy Gradient Methods**.
    
    ### 🎯 Main Focus
    - **Policy Gradient Methods**: REINFORCE and Actor-Critic algorithms
    - **Baseline Techniques**: Variance reduction in policy gradients
    - **Comparison**: Policy-based vs Value-based approaches
    
    ### 🎮 Game Description
    In this space exploration game, the agent rolls dice to explore a dungeon looking for treasures. Each roll has different outcomes:
    - **Treasure (dice 1-3)**: Find treasure (+10 per die)
    - **Minor Risk (dice 4)**: Lose 10% of current treasure
    - **Major Risk (dice 5)**: Lose 30% of current treasure
    - **Game Over (dice 6)**: Lose all treasure
    
    At each turn, the agent must decide:
    - **STOP**: Keep current treasure and end episode
    - **ROLL**: Push luck and roll again (risk vs reward)
    
    ### 🔬 Challenging Variant
    **Experience-based Risk Reduction**: Taking risky decisions builds experience. Each time the agent experiences a negative outcome, that outcome becomes slightly less severe (up to 50% reduction).
    
    ### 🤖 Implemented Agents
    
    1. **REINFORCE (without baseline)**
       - Monte Carlo policy gradient
       - High variance, slower convergence
       - Simple implementation
    
    2. **REINFORCE (with baseline)**
       - Uses running average as baseline
       - Reduced variance, faster convergence
       - Better sample efficiency
    
    3. **Actor-Critic**
       - Separate policy and value networks
       - Online learning (updates after each step)
       - Lower variance than REINFORCE
       - Faster convergence
    
    4. **Q-Learning (baseline)**
       - Value-based approach
       - Tabular Q-values with discretized states
       - ε-greedy exploration
       - For comparison with policy gradient methods
    
    ### 📊 Key Findings
    - **Baseline Effect**: Adding baseline to REINFORCE significantly reduces variance and improves convergence
    - **Actor-Critic**: Generally performs best with online updates and lower variance
    - **Q-Learning**: Competitive performance but requires state discretization
    - **Sample Efficiency**: Policy gradient methods can learn good policies with fewer samples
    
    ### 🛠️ Technical Implementation
    - **Environment**: Custom MDP with stochastic dice outcomes
    - **State Space**: Continuous (treasure, turn, experience)
    - **Action Space**: Discrete (stop, roll)
    - **Neural Networks**: PyTorch for policy and value approximation
    - **Visualization**: Streamlit + Plotly for interactive dashboard
    
    ### 📚 Course Information
    - **Course**: Reinforcement Learning
    - **Instructors**: Prof. Nicolò Cesa-Bianchi, Prof. Alfio Ferrara
    - **Institution**: Università degli Studi di Milano
    - **Program**: Data Science and Economics Master Degree
    - **Academic Year**: 2024-25
    
    ### 🎓 Learning Objectives
    1. Understand policy gradient methods and their advantages
    2. Implement REINFORCE and Actor-Critic algorithms
    3. Analyze the effect of baseline on variance reduction
    4. Compare policy-based and value-based approaches
    5. Design reward structures for exploration-exploitation trade-off
    
    ---
    
    **Made with ❤️ for Reinforcement Learning**
    """)


if __name__ == "__main__":
    main()
