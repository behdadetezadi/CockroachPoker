import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import random
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import pickle


# Import BeliefState from existing agent
from ai_agent import BeliefState


class DQNetwork(nn.Module):
    """
    Deep Q-Network for Cockroach Poker

    Architecture:
    - Input layer: State features
    - Hidden layers: 128 -> 64 -> 32 neurons with ReLU
    - Output layer: 2 Q-values (truth, bluff)
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(DQNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            prev_size = hidden_size

        # Output layer (2 actions: truth, bluff)
        layers.append(nn.Linear(prev_size, 2))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores experiences (state, action, reward, next_state, done)
    and allows random sampling for training.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent with Experience Replay and Target Network

    Key features:
    - Neural network for Q-value approximation
    - Experience replay for stable learning
    - Target network for stable Q-targets
    - Epsilon-greedy exploration
    - POMDP belief state tracking
    """

    def __init__(self, config: Dict, model_path: str = 'dqn_model.pth'):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Hyperparameters
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 100  # Update target network every N steps

        # State representation size
        self.state_size = self._calculate_state_size()

        # Networks
        self.policy_net = DQNetwork(self.state_size).to(self.device)
        self.target_net = DQNetwork(self.state_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # POMDP components
        self.belief_state = BeliefState(config['animals'])

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

        # Action mapping
        self.actions = ['truth', 'bluff']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

        # Load existing model if available
        self.load_model()

    def _calculate_state_size(self) -> int:
        """
        Calculate the size of state representation

        State features:
        - AI hand composition: 8 animals * max count (let's say 8) = 64 (one-hot style)
        - Player face-up: 8 animals * 8 = 64
        - AI face-up: 8 animals * 8 = 64
        - Current claim: 8 (one-hot encoded)
        - Belief probability: 1
        - Recent bluff rate: 1
        - AI has claimed count: 1

        Total: ~195 features (we'll use simpler encoding)

        Simpler encoding:
        - AI hand counts: 8 floats (normalized)
        - Player face-up counts: 8 floats (normalized)
        - AI face-up counts: 8 floats (normalized)
        - Claim one-hot: 8 floats
        - Belief prob: 1 float
        - Bluff rate: 1 float
        - AI has count: 1 float

        Total: 35 features
        """
        num_animals = len(self.config['animals'])

        # 3 count vectors + 1 one-hot + 3 scalars
        state_size = (num_animals * 3) + num_animals + 3

        return state_size

    def state_to_vector(self, state: Dict, belief_state: Optional['BeliefState'] = None) -> np.ndarray:
        """
        Convert game state to neural network input vector

        Args:
            state: Observable game state
            belief_state: Optional belief state (defaults to self.belief_state)

        Returns:
            numpy array of features (normalized)
        """
        belief = belief_state if belief_state is not None else self.belief_state

        num_animals = len(self.config['animals'])
        max_cards = self.config['cards_per_animal']

        features = []

        # AI hand composition (normalized counts)
        ai_hand_counts = defaultdict(int)
        for card in state.get('ai_hand', []):
            ai_hand_counts[card] += 1

        for animal in self.config['animals']:
            features.append(ai_hand_counts[animal] / max_cards)

        # Player face-up cards (normalized counts)
        player_faceup_counts = defaultdict(int)
        for card in state.get('player_face_up', []):
            player_faceup_counts[card] += 1

        for animal in self.config['animals']:
            features.append(player_faceup_counts[animal] / max_cards)

        # AI face-up cards (normalized counts)
        ai_faceup_counts = defaultdict(int)
        for card in state.get('ai_face_up', []):
            ai_faceup_counts[card] += 1

        for animal in self.config['animals']:
            features.append(ai_faceup_counts[animal] / max_cards)

        # Current claim (one-hot encoded)
        claim = state.get('current_claim', None)
        for animal in self.config['animals']:
            features.append(1.0 if claim == animal else 0.0)

        # POMDP features
        if claim and claim != 'none':
            # Belief probability about claim
            truth_prob = belief.get_truth_probability(claim)
            features.append(truth_prob)

            # Recent bluff rate
            bluff_rate = belief.get_recent_bluff_rate()
            features.append(bluff_rate)

            # How many of claimed animal does AI have
            ai_has_claimed = ai_hand_counts.get(claim, 0) / max_cards
            features.append(ai_has_claimed)
        else:
            # No claim yet, use neutral values
            features.append(0.5)  # Neutral belief
            features.append(0.5)  # Neutral bluff rate
            features.append(0.0)  # No claimed card

        return np.array(features, dtype=np.float32)

    def get_observable_state(self, state: Dict) -> Dict:
        """Extract ONLY observable information (no cheating!)"""
        return {
            'ai_hand': state.get('ai_hand', []),
            'player_face_up': state.get('player_face_up', []),
            'ai_face_up': state.get('ai_face_up', []),
            'current_claim': state.get('current_claim', None),
        }

    def choose_response(self, state: Dict) -> str:
        """
        Choose action using epsilon-greedy policy

        Args:
            state: Observable game state

        Returns:
            Action string ('truth' or 'bluff')
        """
        # Extract observable state
        observable_state = self.get_observable_state(state)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation: choose action with highest Q-value
        state_vector = self.state_to_vector(observable_state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()

        return self.idx_to_action[action_idx]

    def choose_card_and_claim(self, hand: List[str]) -> Tuple[str, str]:
        """
        Choose a card to play and what to claim.
        Currently random - could be enhanced with learning.
        """
        if not hand:
            return None, None

        card = random.choice(hand)

        # 50% truth, 50% bluff
        if random.random() < 0.5:
            claim = card
        else:
            claim = random.choice(self.config['animals'])

        return card, claim

    def update_belief(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]):
        """Update belief state after card is revealed"""
        self.belief_state.update_after_reveal(claim, actual_card, cards_remaining)

    def store_experience(self, state: Dict, action: str, reward: float,
                         next_state: Dict, done: bool,
                         belief_state: Optional['BeliefState'] = None,
                         next_belief_state: Optional['BeliefState'] = None):
        """
        Store experience in replay buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            belief_state: Optional belief state for current state
            next_belief_state: Optional belief state for next state
        """
        # Convert to vectors
        state_vector = self.state_to_vector(state, belief_state)
        next_state_vector = self.state_to_vector(next_state, next_belief_state)

        # Convert action to index
        action_idx = self.action_to_idx[action]

        # Store in buffer
        self.replay_buffer.push(state_vector, action_idx, reward, next_state_vector, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def calculate_cards_remaining(self, state: Dict) -> Dict[str, int]:
        """Calculate how many of each animal card are still unaccounted for"""
        cards_remaining = defaultdict(int)

        # Start with total cards per animal
        for animal in self.config['animals']:
            cards_remaining[animal] = self.config['cards_per_animal']

        # Subtract known cards
        for card in state.get('ai_hand', []):
            cards_remaining[card] -= 1
        for card in state.get('player_face_up', []):
            cards_remaining[card] -= 1
        for card in state.get('ai_face_up', []):
            cards_remaining[card] -= 1

        # Ensure non-negative
        for animal in cards_remaining:
            cards_remaining[animal] = max(0, cards_remaining[animal])

        return cards_remaining

    def train_from_logs(self, log_dir: str = 'game_logs', epochs: int = 10) -> Dict:
        """
        Train DQN from logged games

        Args:
            log_dir: Directory containing game logs
            epochs: Number of training epochs

        Returns:
            Dictionary with training statistics
        """
        games = self._load_all_games(log_dir)

        if not games:
            print("No training data found!")
            return {'games': 0, 'experiences': 0}

        print(f"\nLoading experiences from {len(games)} games...")

        # First pass: Load all experiences into replay buffer
        total_experiences = 0

        for game_idx, game in enumerate(games):
            # Reset belief state for each game
            game_belief = BeliefState(self.config['animals'])

            for i, entry in enumerate(game):
                action_data = entry['action']

                # Only train on AI response actions
                if action_data.get('player') == 'ai' and action_data.get('action') == 'respond':
                    state = entry['state']
                    action = action_data['response']
                    reward = entry['reward']
                    is_terminal = state.get('game_over', False)
                    next_state = game[i + 1]['state'] if i + 1 < len(game) else state

                    # Create next_belief by copying and updating
                    next_belief = BeliefState(self.config['animals'])
                    next_belief.opponent_hand_belief = game_belief.opponent_hand_belief.copy()
                    next_belief.truth_count = game_belief.truth_count.copy()
                    next_belief.bluff_count = game_belief.bluff_count.copy()
                    next_belief.claim_history = list(game_belief.claim_history)

                    # Update next_belief with reveal
                    claim = state.get('current_claim')
                    actual = state.get('current_card')
                    if claim and actual:
                        cards_remaining = self.calculate_cards_remaining(state)
                        next_belief.update_after_reveal(claim, actual, cards_remaining)

                    # Get observable states
                    obs_state = self.get_observable_state(state)
                    obs_next = self.get_observable_state(next_state)

                    # Store experience
                    self.store_experience(obs_state, action, reward, obs_next, is_terminal,
                                          game_belief, next_belief)
                    total_experiences += 1

                # Update game_belief for next iteration
                if action_data.get('action') == 'respond':
                    claim = entry['state'].get('current_claim')
                    actual = entry['state'].get('current_card')
                    if claim and actual:
                        cards_remaining = self.calculate_cards_remaining(entry['state'])
                        game_belief.update_after_reveal(claim, actual, cards_remaining)

            if (game_idx + 1) % 5 == 0:
                print(f"  Loaded {game_idx + 1}/{len(games)} games...")

        print(f"✓ Loaded {total_experiences} experiences into replay buffer")
        print(f"\nTraining for {epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Initial epsilon: {self.epsilon:.3f}")
        print("-" * 60)

        # Training loop
        epoch_losses = []

        for epoch in range(epochs):
            epoch_loss = []

            # Train multiple steps per epoch
            steps_per_epoch = max(total_experiences // self.batch_size, 10)

            for step in range(steps_per_epoch):
                loss = self.train_step()
                if loss is not None:
                    epoch_loss.append(loss)

            avg_loss = np.mean(epoch_loss) if epoch_loss else 0
            epoch_losses.append(avg_loss)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {self.epsilon:.3f} | "
                  f"Training Step: {self.training_step}")

        print("-" * 60)
        print("✓ Training complete!")

        return {
            'games': len(games),
            'experiences': total_experiences,
            'epochs': epochs,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'epsilon': self.epsilon
        }

    def _load_all_games(self, log_dir: str) -> List[List[Dict]]:
        """Load all game logs from directory"""
        games = []
        if not os.path.exists(log_dir):
            return games

        for filename in sorted(os.listdir(log_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(log_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        games.append(json.load(f))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return games

    def save_model(self):
        """Save model, optimizer state, and metadata"""
        # Save model and optimizer
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
        }
        torch.save(checkpoint, self.model_path)

        # Save metadata
        metadata = {
            'state_size': self.state_size,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'batch_size': self.batch_size,
            'replay_buffer_size': len(self.replay_buffer),
            'belief_state': self.belief_state.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Deep Q-Network'
        }

        metadata_path = self.model_path.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Model saved: {self.model_path}")
        print(f"  Training steps: {self.training_step}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Replay buffer: {len(self.replay_buffer)} experiences")

    def load_model(self):
        """Load model from disk if exists"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)

                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_step = checkpoint.get('training_step', 0)

                print(f"✓ Model loaded from {self.model_path}")
                print(f"  Training steps: {self.training_step}")
                print(f"  Epsilon: {self.epsilon:.3f}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with fresh model")
        else:
            print("No saved model found. Starting with fresh model")

    def get_statistics(self) -> Dict:
        """Get statistics about the model"""
        stats = {
            'model_type': 'Deep Q-Network',
            'state_size': self.state_size,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'total_parameters': sum(p.numel() for p in self.policy_net.parameters()),
            'device': str(self.device),
            'belief_state': {
                'total_observations': len(self.belief_state.claim_history),
                'truth_counts': dict(self.belief_state.truth_count),
                'bluff_counts': dict(self.belief_state.bluff_count)
            }
        }
        return stats


def train_agent(config: Dict, epochs: int = 10, log_dir: str = 'game_logs'):
    """Standalone training function for DQN agent"""
    print("\n" + "=" * 60)
    print("DEEP Q-NETWORK TRAINING")
    print("=" * 60)
    print("\n✓ Using neural network for Q-value approximation")
    print("✓ Experience replay for stable learning")
    print("✓ Target network for stable Q-targets")
    print("✓ Epsilon-greedy exploration with decay")

    agent = DQNAgent(config)

    print("\nInitial Model Statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        if key != 'belief_state' and not isinstance(value, dict):
            print(f"  {key}: {value}")

    # Train from logs
    training_stats = agent.train_from_logs(log_dir, epochs)

    # Save model
    print("\nSaving trained model...")
    agent.save_model()

    print("\nFinal Model Statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        if key != 'belief_state':
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"  Games processed: {training_stats['games']}")
    print(f"  Experiences collected: {training_stats['experiences']}")
    print(f"  Training epochs: {training_stats['epochs']}")
    print(f"  Final loss: {training_stats['final_loss']:.4f}")
    print(f"  Final epsilon: {training_stats['epsilon']:.3f}")
    print("=" * 60)


def main():
    """Main entry point for training/testing DQN"""
    GAME_CONFIG = {
        'hand_size': 5,
        'animals': ['fly', 'rat', 'toad', 'bat', 'spider', 'cockroach', 'scorpion', 'stinkbug'],
        'lose_threshold': 3,
        'cards_per_animal': 8
    }

    print("\n" + "=" * 60)
    print("COCKROACH POKER - DEEP Q-NETWORK AI")
    print("=" * 60)
    print("\nOptions:")
    print("1. Train DQN agent from existing games")
    print("2. View model statistics")
    print("3. Reset model")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        epochs = input("Enter number of training epochs (default 10): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 10
        train_agent(GAME_CONFIG, epochs=epochs)

    elif choice == '2':
        agent = DQNAgent(GAME_CONFIG)
        print("\nModel Statistics:")
        stats = agent.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    elif choice == '3':
        confirm = input("Delete model? (yes/no): ").strip().lower()
        if confirm == 'yes':
            try:
                os.remove('dqn_model.pth')
                os.remove('dqn_model_metadata.json')
                print("Model deleted!")
            except FileNotFoundError:
                print("No model found.")

    elif choice == '4':
        print("Exiting...")

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()