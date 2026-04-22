import os
import pickle
import random
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from belief import BeliefState
from core import calculate_cards_remaining, load_all_games
from agents.base import Agent


class DQNetwork(nn.Module):
    """3-layer fully-connected network with ReLU + Dropout, Xavier-initialised."""

    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 2))  # 2 actions: truth / bluff
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(Agent):
    """
    Deep Q-Network agent with experience replay and a target network.

    State vector: N*4 + 3 floats  (19 features for 4 animals)
      - AI hand counts (N, normalised)
      - Player face-up counts (N, normalised)
      - AI face-up counts (N, normalised)
      - Claim one-hot (N)
      - Belief probability (1)
      - Recent bluff rate (1)
      - AI-has-claimed count (1)
    """

    ACTIONS = ['truth', 'bluff']

    def __init__(self, config: Dict, model_path: str = 'dqn_model.pth'):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 100

        self.state_size = self._state_size()
        self.policy_net = DQNetwork(self.state_size).to(self.device)
        self.target_net = DQNetwork(self.state_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=10_000)
        self.belief_state = BeliefState(config['animals'])

        self.training_step = 0
        self.action_to_idx = {a: i for i, a in enumerate(self.ACTIONS)}
        self.idx_to_action = {i: a for a, i in self.action_to_idx.items()}

        self.load_model()

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------

    def _state_size(self) -> int:
        n = len(self.config['animals'])
        return n * 4 + 3  # 3 count vecs + 1 one-hot + 3 scalars

    def state_to_vector(self, state: Dict, belief: Optional[BeliefState] = None) -> np.ndarray:
        belief = belief or self.belief_state
        n = len(self.config['animals'])
        max_cards = self.config['cards_per_animal']
        features: List[float] = []

        ai_counts: Dict[str, int] = defaultdict(int)
        for card in state.get('ai_hand', []):
            ai_counts[card] += 1

        p_up: Dict[str, int] = defaultdict(int)
        for card in state.get('player_face_up', []):
            p_up[card] += 1

        a_up: Dict[str, int] = defaultdict(int)
        for card in state.get('ai_face_up', []):
            a_up[card] += 1

        for animal in self.config['animals']:
            features.append(ai_counts[animal] / max_cards)
        for animal in self.config['animals']:
            features.append(p_up[animal] / max_cards)
        for animal in self.config['animals']:
            features.append(a_up[animal] / max_cards)

        claim = state.get('current_claim')
        for animal in self.config['animals']:
            features.append(1.0 if claim == animal else 0.0)

        if claim:
            features.append(belief.get_truth_probability(claim))
            features.append(belief.get_recent_bluff_rate())
            features.append(ai_counts.get(claim, 0) / max_cards)
        else:
            features += [0.5, 0.5, 0.0]

        return np.array(features, dtype=np.float32)

    def get_observable_state(self, state: Dict) -> Dict:
        return {
            'ai_hand': state.get('ai_hand', []),
            'player_face_up': state.get('player_face_up', []),
            'ai_face_up': state.get('ai_face_up', []),
            'current_claim': state.get('current_claim'),
        }

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def choose_response(self, state: Dict) -> str:
        obs = self.get_observable_state(state)

        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)

        vec = self.state_to_vector(obs)
        tensor = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            idx = self.policy_net(tensor).argmax().item()
        return self.idx_to_action[idx]

    def choose_card_and_claim(self, hand: List[str]) -> Tuple[Optional[str], Optional[str]]:
        if not hand:
            return None, None
        card = random.choice(hand)
        claim = card if random.random() < 0.5 else random.choice(self.config['animals'])
        return card, claim

    def update_belief(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]):
        self.belief_state.update_after_reveal(claim, actual_card, cards_remaining)

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def store_experience(
        self,
        state: Dict,
        action: str,
        reward: float,
        next_state: Dict,
        done: bool,
        belief: Optional[BeliefState] = None,
        next_belief: Optional[BeliefState] = None,
    ):
        state_vec = self.state_to_vector(state, belief)
        next_vec = self.state_to_vector(next_state, next_belief)
        self.replay_buffer.push(state_vec, self.action_to_idx[action], reward, next_vec, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.discount_factor * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    # ------------------------------------------------------------------
    # Training from logs
    # ------------------------------------------------------------------

    def train_from_logs(self, log_dir: str = 'game_logs', epochs: int = 10) -> Dict:
        games = load_all_games(log_dir)
        if not games:
            print("No training data found!")
            return {'games': 0, 'experiences': 0}

        print(f"Loading experiences from {len(games)} games...")
        total_experiences = 0

        for game_idx, game in enumerate(games):
            game_belief = BeliefState(self.config['animals'])

            for i, entry in enumerate(game):
                action_data = entry['action']

                if action_data.get('player') == 'ai' and action_data.get('action') == 'respond':
                    state = entry['state']
                    action = action_data['response']
                    reward = entry['reward']
                    is_terminal = state.get('game_over', False)
                    next_state = game[i + 1]['state'] if i + 1 < len(game) else state

                    next_belief = game_belief.copy()
                    claim = state.get('current_claim')
                    actual = state.get('current_card')
                    if claim and actual:
                        remaining = calculate_cards_remaining(state, self.config)
                        next_belief.update_after_reveal(claim, actual, remaining)

                    obs = self.get_observable_state(state)
                    obs_next = self.get_observable_state(next_state)
                    self.store_experience(obs, action, reward, obs_next, is_terminal,
                                         game_belief, next_belief)
                    total_experiences += 1

                if action_data.get('action') == 'respond':
                    claim = entry['state'].get('current_claim')
                    actual = entry['state'].get('current_card')
                    if claim and actual:
                        remaining = calculate_cards_remaining(entry['state'], self.config)
                        game_belief.update_after_reveal(claim, actual, remaining)

            if (game_idx + 1) % 5 == 0:
                print(f"  Loaded {game_idx + 1}/{len(games)} games...")

        print(f"Loaded {total_experiences} experiences into replay buffer")
        print(f"Training for {epochs} epochs (batch={self.batch_size}, eps={self.epsilon:.3f})")

        epoch_losses = []
        for epoch in range(epochs):
            steps = max(total_experiences // self.batch_size, 10)
            losses = [self.train_step() for _ in range(steps)]
            losses = [l for l in losses if l is not None]
            avg = float(np.mean(losses)) if losses else 0.0
            epoch_losses.append(avg)
            print(f"Epoch {epoch+1}/{epochs} | loss={avg:.4f} | eps={self.epsilon:.3f} | step={self.training_step}")

        return {
            'games': len(games),
            'experiences': total_experiences,
            'epochs': epochs,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'epsilon': self.epsilon,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
        }
        torch.save(checkpoint, self.model_path)

        import json
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
            'model_type': 'Deep Q-Network',
        }
        meta_path = self.model_path.replace('.pth', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved: {self.model_path}  (step={self.training_step}, eps={self.epsilon:.3f})")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
                self.target_net.load_state_dict(ckpt['target_net_state_dict'])
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.epsilon = ckpt.get('epsilon', self.epsilon)
                self.training_step = ckpt.get('training_step', 0)
                print(f"Model loaded: {self.model_path}  (step={self.training_step}, eps={self.epsilon:.3f})")
            except Exception as e:
                print(f"Error loading model: {e} — starting fresh")
        else:
            print("No saved model found. Starting with fresh model")

    def get_statistics(self) -> Dict:
        return {
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
                'bluff_counts': dict(self.belief_state.bluff_count),
            },
        }
