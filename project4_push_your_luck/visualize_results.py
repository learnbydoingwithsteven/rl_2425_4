"""
Standalone visualization script for training results
Creates matplotlib plots showing training progress and comparisons
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_results():
    """Load all training results."""
    results = {}
    results_dir = "results"
    
    files = {
        "REINFORCE\n(no baseline)": "reinforce_no_baseline.json",
        "REINFORCE\n(baseline)": "reinforce_baseline.json",
        "Actor-Critic": "actor_critic.json",
        "Q-Learning": "qlearning.json"
    }
    
    for name, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[name] = json.load(f)
    
    return results

def plot_training_curves(results):
    """Plot training curves for all agents."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Push Your Luck! - Training Progress Comparison', fontsize=16, fontweight='bold')
    
    colors = {
        "REINFORCE\n(no baseline)": "#FF6B6B",
        "REINFORCE\n(baseline)": "#4ECDC4",
        "Actor-Critic": "#45B7D1",
        "Q-Learning": "#FFA07A"
    }
    
    # Plot 1: Raw Returns
    ax1 = axes[0, 0]
    for agent_name, history in results.items():
        color = colors.get(agent_name, "#888888")
        ax1.plot(history["returns"], label=agent_name, color=color, alpha=0.3, linewidth=1)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Return', fontsize=12)
    ax1.set_title('Episode Returns (Raw)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Average Returns
    ax2 = axes[0, 1]
    window = 50
    for agent_name, history in results.items():
        color = colors.get(agent_name, "#888888")
        if len(history["returns"]) >= window:
            ma_returns = pd.Series(history["returns"]).rolling(window=window).mean()
            ax2.plot(ma_returns, label=agent_name, color=color, linewidth=2.5)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Return (Moving Average)', fontsize=12)
    ax2.set_title(f'Moving Average Returns (window={window})', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode Lengths
    ax3 = axes[1, 0]
    for agent_name, history in results.items():
        color = colors.get(agent_name, "#888888")
        ma_lengths = pd.Series(history["episode_lengths"]).rolling(window=50).mean()
        ax3.plot(ma_lengths, label=agent_name, color=color, linewidth=2)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Episode Length (MA)', fontsize=12)
    ax3.set_title('Episode Lengths (Moving Average)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Treasures
    ax4 = axes[1, 1]
    for agent_name, history in results.items():
        color = colors.get(agent_name, "#888888")
        ma_treasures = pd.Series(history["treasures"]).rolling(window=50).mean()
        ax4.plot(ma_treasures, label=agent_name, color=color, linewidth=2)
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Treasure (MA)', fontsize=12)
    ax4.set_title('Final Treasures (Moving Average)', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: training_curves.png")
    plt.show()

def plot_performance_comparison(results):
    """Plot performance comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Push Your Luck! - Performance Comparison', fontsize=16, fontweight='bold')
    
    agent_names = list(results.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    
    # Final Returns
    final_returns = [np.mean(results[name]["returns"][-100:]) for name in agent_names]
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(agent_names)), final_returns, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(agent_names)))
    ax1.set_xticklabels(agent_names, rotation=15, ha='right')
    ax1.set_ylabel('Average Return', fontsize=12)
    ax1.set_title('Average Return\n(last 100 episodes)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, final_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Final Treasures
    final_treasures = [np.mean(results[name]["treasures"][-100:]) for name in agent_names]
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(agent_names)), final_treasures, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(agent_names)))
    ax2.set_xticklabels(agent_names, rotation=15, ha='right')
    ax2.set_ylabel('Average Treasure', fontsize=12)
    ax2.set_title('Average Treasure\n(last 100 episodes)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars2, final_treasures):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Training Time
    training_times = [results[name]["training_time"] for name in agent_names]
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(agent_names)), training_times, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(agent_names)))
    ax3.set_xticklabels(agent_names, rotation=15, ha='right')
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Training Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, training_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: performance_comparison.png")
    plt.show()

def plot_baseline_effect(results):
    """Plot effect of baseline on REINFORCE."""
    if "REINFORCE\n(no baseline)" not in results or "REINFORCE\n(baseline)" not in results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Effect of Baseline on REINFORCE Performance', fontsize=16, fontweight='bold')
    
    # No baseline
    no_baseline = results["REINFORCE\n(no baseline)"]
    ma_no_baseline = pd.Series(no_baseline["returns"]).rolling(window=50).mean()
    ax.plot(ma_no_baseline, label="REINFORCE (no baseline)", color="#FF6B6B", linewidth=3)
    
    # With baseline
    baseline = results["REINFORCE\n(baseline)"]
    ma_baseline = pd.Series(baseline["returns"]).rolling(window=50).mean()
    ax.plot(ma_baseline, label="REINFORCE (with baseline)", color="#4ECDC4", linewidth=3)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Moving Average Return (window=50)', fontsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_effect.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: baseline_effect.png")
    plt.show()

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TRAINING RESULTS SUMMARY")
    print("="*70)
    
    for agent_name, history in results.items():
        print(f"\n{agent_name}:")
        print(f"  Episodes: {len(history['returns'])}")
        print(f"  Avg Return (last 100): {np.mean(history['returns'][-100:]):.2f}")
        print(f"  Max Return: {np.max(history['returns']):.2f}")
        print(f"  Avg Treasure (last 100): {np.mean(history['treasures'][-100:]):.2f}")
        print(f"  Avg Episode Length (last 100): {np.mean(history['episode_lengths'][-100:]):.2f}")
        print(f"  Training Time: {history['training_time']:.2f}s")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Find best performer
    best_agent = max(results.keys(), key=lambda x: np.mean(results[x]['returns'][-100:]))
    best_return = np.mean(results[best_agent]['returns'][-100:])
    
    print(f"\n🏆 Best Performer: {best_agent}")
    print(f"   Average Return: {best_return:.2f}")
    
    # Compare REINFORCE variants
    if "REINFORCE\n(no baseline)" in results and "REINFORCE\n(baseline)" in results:
        no_base = np.mean(results["REINFORCE\n(no baseline)"]["returns"][-100:])
        with_base = np.mean(results["REINFORCE\n(baseline)"]["returns"][-100:])
        improvement = ((with_base - no_base) / (abs(no_base) + 1e-8)) * 100
        
        print(f"\n📊 Baseline Effect on REINFORCE:")
        print(f"   Without baseline: {no_base:.2f}")
        print(f"   With baseline: {with_base:.2f}")
        print(f"   Improvement: {improvement:.1f}%")
    
    # Compare policy vs value-based
    if "Actor-Critic" in results and "Q-Learning" in results:
        ac_return = np.mean(results["Actor-Critic"]["returns"][-100:])
        ql_return = np.mean(results["Q-Learning"]["returns"][-100:])
        
        print(f"\n🤖 Policy Gradient vs Value-Based:")
        print(f"   Actor-Critic: {ac_return:.2f}")
        print(f"   Q-Learning: {ql_return:.2f}")
        print(f"   Difference: {ac_return - ql_return:.2f}")
    
    print("\n" + "="*70)

def main():
    """Main visualization function."""
    print("\n🎲 Push Your Luck! - Results Visualization\n")
    
    # Load results
    print("Loading training results...")
    results = load_results()
    
    if not results:
        print("❌ No training results found. Please run training first.")
        return
    
    print(f"✅ Loaded results for {len(results)} agents\n")
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    print("\n📊 Creating visualizations...\n")
    
    plot_training_curves(results)
    plot_performance_comparison(results)
    plot_baseline_effect(results)
    
    print("\n✅ All visualizations created successfully!")
    print("\n📁 Saved files:")
    print("   - training_curves.png")
    print("   - performance_comparison.png")
    print("   - baseline_effect.png")
    print("\n🚀 To launch interactive dashboard, run: streamlit run app.py")

if __name__ == "__main__":
    main()
