"""
RL Agents for Push Your Luck!
Implements: REINFORCE, Actor-Critic, Q-Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
from typing import List, Tuple, Dict
import random


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Neural network for value function approximation."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent.
    Uses complete episode returns for policy updates.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 use_baseline: bool = True):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for policy network
            gamma: Discount factor
            use_baseline: Whether to use baseline (average return) to reduce variance
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode memory
        self.log_probs = []
        self.rewards = []
        
        # Baseline (running average of returns)
        self.baseline = 0
        self.baseline_alpha = 0.1
        
        # Training statistics
        self.episode_returns = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode (stochastic) or evaluation (greedy)
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if training:
            # Forward pass with gradients
            action_probs = self.policy(state_tensor)
            
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            # Store log probability for training (with gradients)
            self.log_probs.append(dist.log_prob(action))
            
            return action.item()
        else:
            # Greedy action without gradients
            with torch.no_grad():
                action_probs = self.policy(state_tensor)
            return torch.argmax(action_probs).item()
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using REINFORCE algorithm.
        Called at end of episode.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.rewards) == 0:
            return {"loss": 0, "return": 0}
        
        # Calculate discounted returns (G_t)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        episode_return = returns[0].item()
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Apply baseline if enabled
        if self.use_baseline:
            # Update baseline (running average)
            self.baseline = self.baseline_alpha * episode_return + (1 - self.baseline_alpha) * self.baseline
            advantages = returns - self.baseline
        else:
            advantages = returns
        
        # Calculate policy gradient loss
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            # Detach advantage to avoid backprop through it
            policy_loss.append(-log_prob * advantage.detach())
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Store statistics
        self.episode_returns.append(episode_return)
        
        # Clear episode memory
        metrics = {
            "loss": policy_loss.item(),
            "return": episode_return,
            "baseline": self.baseline if self.use_baseline else 0
        }
        
        self.log_probs = []
        self.rewards = []
        
        return metrics


class ActorCriticAgent:
    """
    Actor-Critic Agent with separate policy and value networks.
    Updates after each step (online learning).
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 actor_lr: float = 0.001, critic_lr: float = 0.005,
                 gamma: float = 0.99):
        """
        Initialize Actor-Critic agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_lr: Learning rate for actor (policy)
            critic_lr: Learning rate for critic (value function)
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Actor (policy) network
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic (value) network
        self.critic = ValueNetwork(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Training statistics
        self.episode_returns = []
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode
            
        Returns:
            (action, log_prob) tuple
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        
        if training:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob
        else:
            return torch.argmax(action_probs).item(), None
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool, log_prob: torch.Tensor) -> Dict[str, float]:
        """
        Update actor and critic networks using TD error.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            
        Returns:
            Dictionary with training metrics
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Compute value estimates
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor) if not done else torch.zeros(1, 1)
        
        # Compute TD error (advantage)
        td_target = reward + self.gamma * next_value
        td_error = td_target - value
        
        # Update critic
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -log_prob * td_error.detach()  # Detach to avoid backprop through critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": td_error.item(),
            "value": value.item()
        }


class QLearningAgent:
    """
    Q-Learning Agent (value-based baseline).
    Uses tabular Q-values with discretized state space.
    """
    
    def __init__(self, action_dim: int, learning_rate: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            action_dim: Number of actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
        """
        self.action_dim = action_dim
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table (state -> action -> value)
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        
        # Training statistics
        self.episode_returns = []
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state into bins for tabular representation.
        
        Args:
            state: State vector [treasure, turn_ratio, can_continue, ...]
            
        Returns:
            Discretized state tuple
        """
        # Discretize treasure into bins
        treasure = int(state[0] * 100)  # Denormalize
        treasure_bin = min(treasure // 20, 10)  # 0-10 bins
        
        # Discretize turn ratio
        turn_ratio = state[1]
        turn_bin = int(turn_ratio * 10)  # 0-10 bins
        
        # Can continue flag
        can_continue = int(state[2])
        
        return (treasure_bin, turn_bin, can_continue)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        state_key = self._discretize_state(state)
        
        if training and random.random() < self.epsilon:
            # Explore
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary with training metrics
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.q_table[next_state_key]
            target_q = reward + self.gamma * np.max(next_q_values)
        
        # Q-learning update
        td_error = target_q - current_q
        self.q_table[state_key][action] += self.alpha * td_error
        
        return {
            "td_error": td_error,
            "q_value": current_q,
            "target_q": target_q
        }
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
