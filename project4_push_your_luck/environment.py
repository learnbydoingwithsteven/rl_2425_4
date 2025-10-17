"""
Push Your Luck! - Space Exploration Dice Game Environment
MDP Implementation for Project 4
"""

import numpy as np
from typing import Tuple, Dict, List
import random


class DiceExplorationEnv:
    """
    Space exploration game where agent rolls dice to find treasures.
    Each roll has outcomes affecting treasury and risks (injury, game over, loss).
    Agent decides: stop and keep treasures, or push luck and roll again.
    """
    
    def __init__(self, num_dice: int = 3, max_turns: int = 10, 
                 enable_learning_variant: bool = False):
        """
        Initialize the environment.
        
        Args:
            num_dice: Number of dice to roll (default: 3)
            max_turns: Maximum turns per episode (default: 10)
            enable_learning_variant: Enable experience-based risk reduction
        """
        self.num_dice = num_dice
        self.max_turns = max_turns
        self.enable_learning_variant = enable_learning_variant
        
        # Dice outcomes (1-6 on each die)
        self.dice_faces = 6
        
        # Define dice outcome meanings
        self.outcome_types = {
            1: "treasure",      # Find treasure
            2: "treasure",      # Find treasure
            3: "treasure",      # Find treasure
            4: "minor_risk",    # Minor injury (-10% current treasure)
            5: "major_risk",    # Major injury (-30% current treasure)
            6: "game_over"      # Game over (lose all)
        }
        
        # Treasure values per die showing treasure
        self.treasure_per_die = 10
        
        # Risk penalties
        self.minor_risk_penalty = 0.10  # 10% loss
        self.major_risk_penalty = 0.30  # 30% loss
        
        # Experience tracking for learning variant
        self.experience_counts = {
            "minor_risk": 0,
            "major_risk": 0,
            "game_over": 0
        }
        
        # Learning rate for risk reduction
        self.risk_learning_rate = 0.02  # 2% reduction per experience
        
        self.reset()
    
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.current_treasure = 0
        self.turn = 0
        self.done = False
        self.total_rolls = 0
        
        # Reset experience if not using learning variant
        if not self.enable_learning_variant:
            self.experience_counts = {
                "minor_risk": 0,
                "major_risk": 0,
                "game_over": 0
            }
        
        return self._get_state()
    
    def _get_state(self) -> Dict:
        """Get current state representation."""
        return {
            "treasure": self.current_treasure,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "done": self.done,
            "experience": self.experience_counts.copy() if self.enable_learning_variant else None
        }
    
    def _roll_dice(self) -> List[int]:
        """Roll all dice and return results."""
        return [random.randint(1, self.dice_faces) for _ in range(self.num_dice)]
    
    def _get_adjusted_penalty(self, risk_type: str, base_penalty: float) -> float:
        """
        Calculate adjusted penalty based on experience (learning variant).
        
        Args:
            risk_type: Type of risk experienced
            base_penalty: Base penalty percentage
            
        Returns:
            Adjusted penalty percentage
        """
        if not self.enable_learning_variant:
            return base_penalty
        
        # Reduce penalty based on experience
        experience = self.experience_counts.get(risk_type, 0)
        reduction = min(experience * self.risk_learning_rate, base_penalty * 0.5)  # Max 50% reduction
        return max(base_penalty - reduction, base_penalty * 0.5)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 = stop and keep treasure, 1 = roll dice again
            
        Returns:
            state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0, True, {"message": "Episode already finished"}
        
        info = {}
        
        # Action 0: Stop and keep treasure
        if action == 0:
            reward = self.current_treasure
            self.done = True
            info["message"] = f"Stopped! Kept {self.current_treasure} treasure."
            return self._get_state(), reward, True, info
        
        # Action 1: Roll dice
        self.turn += 1
        self.total_rolls += 1
        
        if self.turn >= self.max_turns:
            # Forced to stop at max turns
            reward = self.current_treasure
            self.done = True
            info["message"] = f"Max turns reached! Final treasure: {self.current_treasure}"
            return self._get_state(), reward, True, info
        
        # Roll dice
        dice_results = self._roll_dice()
        info["dice_results"] = dice_results
        
        # Count outcomes
        outcome_counts = {
            "treasure": 0,
            "minor_risk": 0,
            "major_risk": 0,
            "game_over": 0
        }
        
        for die in dice_results:
            outcome = self.outcome_types[die]
            outcome_counts[outcome] += 1
        
        info["outcomes"] = outcome_counts
        
        # Process outcomes
        reward = 0
        
        # Check for game over first
        if outcome_counts["game_over"] > 0:
            # Game over - lose all treasure
            lost_treasure = self.current_treasure
            self.current_treasure = 0
            reward = -lost_treasure if lost_treasure > 0 else -10  # Penalty for game over
            self.done = True
            
            # Track experience
            self.experience_counts["game_over"] += outcome_counts["game_over"]
            
            info["message"] = f"GAME OVER! Lost all {lost_treasure} treasure!"
            return self._get_state(), reward, True, info
        
        # Process major risks
        if outcome_counts["major_risk"] > 0:
            adjusted_penalty = self._get_adjusted_penalty("major_risk", self.major_risk_penalty)
            loss = int(self.current_treasure * adjusted_penalty * outcome_counts["major_risk"])
            self.current_treasure -= loss
            reward -= loss
            self.experience_counts["major_risk"] += outcome_counts["major_risk"]
            info["major_risk_loss"] = loss
            info["adjusted_major_penalty"] = adjusted_penalty
        
        # Process minor risks
        if outcome_counts["minor_risk"] > 0:
            adjusted_penalty = self._get_adjusted_penalty("minor_risk", self.minor_risk_penalty)
            loss = int(self.current_treasure * adjusted_penalty * outcome_counts["minor_risk"])
            self.current_treasure -= loss
            reward -= loss
            self.experience_counts["minor_risk"] += outcome_counts["minor_risk"]
            info["minor_risk_loss"] = loss
            info["adjusted_minor_penalty"] = adjusted_penalty
        
        # Add treasure
        treasure_gained = outcome_counts["treasure"] * self.treasure_per_die
        self.current_treasure += treasure_gained
        reward += treasure_gained
        info["treasure_gained"] = treasure_gained
        
        # Ensure treasure doesn't go negative
        self.current_treasure = max(0, self.current_treasure)
        
        info["message"] = f"Turn {self.turn}: Gained {treasure_gained}, Current: {self.current_treasure}"
        
        return self._get_state(), reward, False, info
    
    def get_state_vector(self, state: Dict) -> np.ndarray:
        """
        Convert state dict to vector for neural networks.
        
        Returns:
            State vector: [treasure, turn, turn_ratio, experience_features...]
        """
        features = [
            state["treasure"] / 100.0,  # Normalize treasure
            state["turn"] / self.max_turns,  # Turn ratio
            1.0 if state["turn"] < self.max_turns else 0.0,  # Can continue
        ]
        
        # Add experience features if learning variant enabled
        if self.enable_learning_variant and state["experience"]:
            features.extend([
                state["experience"]["minor_risk"] / 10.0,
                state["experience"]["major_risk"] / 10.0,
                state["experience"]["game_over"] / 10.0,
            ])
        
        return np.array(features, dtype=np.float32)
    
    @property
    def state_dim(self) -> int:
        """Get state vector dimension."""
        base_dim = 3
        if self.enable_learning_variant:
            base_dim += 3  # Experience features
        return base_dim
    
    @property
    def action_dim(self) -> int:
        """Get action space dimension."""
        return 2  # Stop or Roll


if __name__ == "__main__":
    # Test the environment
    print("Testing Push Your Luck Environment\n")
    
    # Test basic version
    print("=== Basic Version ===")
    env = DiceExplorationEnv(num_dice=3, max_turns=10, enable_learning_variant=False)
    state = env.reset()
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        state = env.reset()
        total_reward = 0
        
        while not state["done"]:
            # Random policy for testing
            action = random.choice([0, 1])
            action_name = "STOP" if action == 0 else "ROLL"
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"  Action: {action_name}, Reward: {reward:.1f}, {info['message']}")
            
            if done:
                break
        
        print(f"  Total Reward: {total_reward:.1f}")
    
    # Test learning variant
    print("\n\n=== Learning Variant (Experience-based Risk Reduction) ===")
    env = DiceExplorationEnv(num_dice=3, max_turns=10, enable_learning_variant=True)
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        state = env.reset()
        total_reward = 0
        
        while not state["done"]:
            action = random.choice([0, 1])
            action_name = "STOP" if action == 0 else "ROLL"
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"  Action: {action_name}, Reward: {reward:.1f}, {info['message']}")
            if "adjusted_minor_penalty" in info:
                print(f"    Adjusted minor penalty: {info['adjusted_minor_penalty']:.2%}")
            if "adjusted_major_penalty" in info:
                print(f"    Adjusted major penalty: {info['adjusted_major_penalty']:.2%}")
            
            if done:
                break
        
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Experience: {state['experience']}")
