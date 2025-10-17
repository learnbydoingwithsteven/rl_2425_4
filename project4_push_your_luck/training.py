"""
Training script for Push Your Luck! agents
Trains REINFORCE, Actor-Critic, and Q-Learning agents
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os
from tqdm import tqdm
import time

from environment import DiceExplorationEnv
from agents import REINFORCEAgent, ActorCriticAgent, QLearningAgent


class Trainer:
    """Trainer class for RL agents."""
    
    def __init__(self, env: DiceExplorationEnv, save_dir: str = "results"):
        """
        Initialize trainer.
        
        Args:
            env: Environment instance
            save_dir: Directory to save results
        """
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.training_history = {}
    
    def train_reinforce(self, num_episodes: int = 1000, 
                       use_baseline: bool = True) -> Dict:
        """
        Train REINFORCE agent.
        
        Args:
            num_episodes: Number of training episodes
            use_baseline: Whether to use baseline
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training REINFORCE Agent (baseline={use_baseline})")
        print(f"{'='*60}")
        
        agent = REINFORCEAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            use_baseline=use_baseline
        )
        
        history = {
            "returns": [],
            "losses": [],
            "baselines": [],
            "episode_lengths": [],
            "treasures": []
        }
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="REINFORCE Training"):
            state_dict = self.env.reset()
            state = self.env.get_state_vector(state_dict)
            episode_length = 0
            
            # Run episode
            while not state_dict["done"]:
                action = agent.select_action(state, training=True)
                state_dict, reward, done, info = self.env.step(action)
                next_state = self.env.get_state_vector(state_dict)
                
                agent.store_reward(reward)
                
                state = next_state
                episode_length += 1
            
            # Update policy at end of episode
            metrics = agent.update()
            
            # Store metrics
            history["returns"].append(metrics["return"])
            history["losses"].append(metrics["loss"])
            history["baselines"].append(metrics["baseline"])
            history["episode_lengths"].append(episode_length)
            history["treasures"].append(state_dict["treasure"])
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_return = np.mean(history["returns"][-100:])
                avg_treasure = np.mean(history["treasures"][-100:])
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Avg Return (last 100): {avg_return:.2f}")
                print(f"  Avg Treasure (last 100): {avg_treasure:.2f}")
                print(f"  Baseline: {metrics['baseline']:.2f}")
        
        training_time = time.time() - start_time
        history["training_time"] = training_time
        history["agent_type"] = f"REINFORCE (baseline={use_baseline})"
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final avg return (last 100): {np.mean(history['returns'][-100:]):.2f}")
        
        return history
    
    def train_actor_critic(self, num_episodes: int = 1000) -> Dict:
        """
        Train Actor-Critic agent.
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training Actor-Critic Agent")
        print(f"{'='*60}")
        
        agent = ActorCriticAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        history = {
            "returns": [],
            "actor_losses": [],
            "critic_losses": [],
            "td_errors": [],
            "episode_lengths": [],
            "treasures": []
        }
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Actor-Critic Training"):
            state_dict = self.env.reset()
            state = self.env.get_state_vector(state_dict)
            episode_return = 0
            episode_length = 0
            
            # Run episode
            while not state_dict["done"]:
                action, log_prob = agent.select_action(state, training=True)
                state_dict, reward, done, info = self.env.step(action)
                next_state = self.env.get_state_vector(state_dict)
                
                # Update after each step
                metrics = agent.update(state, action, reward, next_state, done, log_prob)
                
                episode_return += reward
                state = next_state
                episode_length += 1
            
            # Store metrics
            history["returns"].append(episode_return)
            history["episode_lengths"].append(episode_length)
            history["treasures"].append(state_dict["treasure"])
            
            # Store average losses for this episode
            if len(agent.actor_losses) > 0:
                history["actor_losses"].append(np.mean(agent.actor_losses[-episode_length:]))
                history["critic_losses"].append(np.mean(agent.critic_losses[-episode_length:]))
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_return = np.mean(history["returns"][-100:])
                avg_treasure = np.mean(history["treasures"][-100:])
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Avg Return (last 100): {avg_return:.2f}")
                print(f"  Avg Treasure (last 100): {avg_treasure:.2f}")
        
        training_time = time.time() - start_time
        history["training_time"] = training_time
        history["agent_type"] = "Actor-Critic"
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final avg return (last 100): {np.mean(history['returns'][-100:]):.2f}")
        
        return history
    
    def train_qlearning(self, num_episodes: int = 1000) -> Dict:
        """
        Train Q-Learning agent.
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training Q-Learning Agent")
        print(f"{'='*60}")
        
        agent = QLearningAgent(action_dim=self.env.action_dim)
        
        history = {
            "returns": [],
            "td_errors": [],
            "epsilons": [],
            "episode_lengths": [],
            "treasures": []
        }
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Q-Learning Training"):
            state_dict = self.env.reset()
            state = self.env.get_state_vector(state_dict)
            episode_return = 0
            episode_length = 0
            
            # Run episode
            while not state_dict["done"]:
                action = agent.select_action(state, training=True)
                state_dict, reward, done, info = self.env.step(action)
                next_state = self.env.get_state_vector(state_dict)
                
                # Update Q-values
                metrics = agent.update(state, action, reward, next_state, done)
                
                episode_return += reward
                state = next_state
                episode_length += 1
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Store metrics
            history["returns"].append(episode_return)
            history["epsilons"].append(agent.epsilon)
            history["episode_lengths"].append(episode_length)
            history["treasures"].append(state_dict["treasure"])
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_return = np.mean(history["returns"][-100:])
                avg_treasure = np.mean(history["treasures"][-100:])
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Avg Return (last 100): {avg_return:.2f}")
                print(f"  Avg Treasure (last 100): {avg_treasure:.2f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        history["training_time"] = training_time
        history["agent_type"] = "Q-Learning"
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final avg return (last 100): {np.mean(history['returns'][-100:]):.2f}")
        
        return history
    
    def save_results(self, history: Dict, filename: str):
        """Save training results to JSON."""
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, (list, np.ndarray)):
                serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                            for v in value]
            else:
                serializable_history[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Results saved to {filepath}")


def main():
    """Main training script."""
    
    # Create environment
    print("Initializing environment...")
    env = DiceExplorationEnv(
        num_dice=3,
        max_turns=10,
        enable_learning_variant=False  # Start with basic version
    )
    
    # Create trainer
    trainer = Trainer(env, save_dir="results")
    
    # Training parameters
    num_episodes = 1000
    
    # Train all agents
    print("\n" + "="*60)
    print("TRAINING ALL AGENTS")
    print("="*60)
    
    # 1. REINFORCE without baseline
    history_reinforce_no_baseline = trainer.train_reinforce(
        num_episodes=num_episodes,
        use_baseline=False
    )
    trainer.save_results(history_reinforce_no_baseline, "reinforce_no_baseline.json")
    
    # 2. REINFORCE with baseline
    history_reinforce_baseline = trainer.train_reinforce(
        num_episodes=num_episodes,
        use_baseline=True
    )
    trainer.save_results(history_reinforce_baseline, "reinforce_baseline.json")
    
    # 3. Actor-Critic
    history_actor_critic = trainer.train_actor_critic(num_episodes=num_episodes)
    trainer.save_results(history_actor_critic, "actor_critic.json")
    
    # 4. Q-Learning
    history_qlearning = trainer.train_qlearning(num_episodes=num_episodes)
    trainer.save_results(history_qlearning, "qlearning.json")
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETED!")
    print("="*60)
    
    # Summary
    print("\nTraining Summary:")
    print(f"REINFORCE (no baseline): {np.mean(history_reinforce_no_baseline['returns'][-100:]):.2f}")
    print(f"REINFORCE (baseline):    {np.mean(history_reinforce_baseline['returns'][-100:]):.2f}")
    print(f"Actor-Critic:            {np.mean(history_actor_critic['returns'][-100:]):.2f}")
    print(f"Q-Learning:              {np.mean(history_qlearning['returns'][-100:]):.2f}")


if __name__ == "__main__":
    main()
