"""
Quick training script for demonstration (500 episodes per agent)
"""

import numpy as np
import os
from environment import DiceExplorationEnv
from training import Trainer

def main():
    print("="*60)
    print("QUICK TRAINING - Push Your Luck!")
    print("Training 4 agents with 500 episodes each")
    print("="*60)
    
    # Create environment
    env = DiceExplorationEnv(
        num_dice=3,
        max_turns=10,
        enable_learning_variant=False
    )
    
    # Create trainer
    trainer = Trainer(env, save_dir="results")
    
    num_episodes = 500
    
    # Train all agents
    print("\n1/4: Training REINFORCE (no baseline)...")
    history1 = trainer.train_reinforce(num_episodes=num_episodes, use_baseline=False)
    trainer.save_results(history1, "reinforce_no_baseline.json")
    
    print("\n2/4: Training REINFORCE (with baseline)...")
    history2 = trainer.train_reinforce(num_episodes=num_episodes, use_baseline=True)
    trainer.save_results(history2, "reinforce_baseline.json")
    
    print("\n3/4: Training Actor-Critic...")
    history3 = trainer.train_actor_critic(num_episodes=num_episodes)
    trainer.save_results(history3, "actor_critic.json")
    
    print("\n4/4: Training Q-Learning...")
    history4 = trainer.train_qlearning(num_episodes=num_episodes)
    trainer.save_results(history4, "qlearning.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # Summary
    print("\nFinal Performance (Avg Return, last 100 episodes):")
    print(f"  REINFORCE (no baseline): {np.mean(history1['returns'][-100:]):.2f}")
    print(f"  REINFORCE (baseline):    {np.mean(history2['returns'][-100:]):.2f}")
    print(f"  Actor-Critic:            {np.mean(history3['returns'][-100:]):.2f}")
    print(f"  Q-Learning:              {np.mean(history4['returns'][-100:]):.2f}")
    
    print("\nResults saved to 'results/' directory")
    print("Run 'streamlit run app.py' to view interactive dashboard!")

if __name__ == "__main__":
    main()
