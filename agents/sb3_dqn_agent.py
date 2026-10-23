import os
import random

import numpy as np

from agents.base import Agent
from belief import BeliefState
from core import GAME_CONFIG, calculate_cards_remaining, load_all_games
from env import state_to_obs, CockroachPokerEnv


class SB3DQNAgent(Agent):
    """
    DQN agent using stable-baselines3 network infrastructure, trained offline
    from human game logs — not from self-play.

    Why offline from logs:
      The goal is to learn one specific person's bluffing patterns.
      Self-play against a random or varied opponent cannot capture that.
      Only games against the real person contain their behavioral signal.

    Training source:  game_logs/*.json  (same as POMDP agent)
    Inference:        SB3's q_net, via the standard Agent interface
    Gradient steps:   run manually on SB3's q_net / q_net_target so we
                      control the data source without fighting SB3's learn() loop
    """

    ACTIONS = ['truth', 'bluff']

    def __init__(self, config: dict | None = None, model_path: str = 'sb3_dqn'):
        self.config = config or GAME_CONFIG
        self.model_path = model_path
        self.belief_state = BeliefState(self.config['animals'])
        self.model = None
        self._n_grad_steps = 0
        self._load_model()

    # ------------------------------------------------------------------
    # Agent interface  (called by ui.py during a live game)
    # ------------------------------------------------------------------

    def choose_response(self, state: dict) -> str:
        if self.model is None:
            return random.choice(self.ACTIONS)
        obs = state_to_obs(state, self.config, self.belief_state)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.ACTIONS[int(action)]

    def choose_card_and_claim(self, hand: list[str]) -> tuple[str | None, str | None]:
        if not hand:
            return None, None
        card = random.choice(hand)
        claim = card if random.random() < 0.5 else random.choice(self.config['animals'])
        return card, claim

    def update_belief(self, claim: str, actual_card: str, cards_remaining: dict):
        self.belief_state.update_after_reveal(claim, actual_card, cards_remaining)

    # ------------------------------------------------------------------
    # Offline training from human game logs
    # ------------------------------------------------------------------

    def train_from_logs(self, log_dir: str = 'game_logs', epochs: int = 10) -> dict:
        """
        Load the person's game logs, build (obs, action, reward, next_obs, done)
        tuples, then run gradient descent directly on SB3's q_net.

        This gives us SB3's cleaner network architecture (MlpPolicy, target net,
        Huber loss, gradient clipping) while keeping the training data sourced
        entirely from the real human opponent.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            print("PyTorch not installed (it is required by stable-baselines3).")
            return {'games': 0, 'experiences': 0}

        games = load_all_games(log_dir)
        if not games:
            print("No game logs found. Play some games first (option 1 in main.py).")
            return {'games': 0, 'experiences': 0}

        # Create a fresh model if none is loaded
        if self.model is None:
            self.model = self._make_model()

        # ---- Build experience list from logs -------------------------
        print(f"Loading experiences from {len(games)} games...")
        experiences = []  # (obs, action_idx, reward, next_obs, done)

        for game in games:
            game_belief = BeliefState(self.config['animals'])

            for i, entry in enumerate(game):
                action_data = entry['action']

                if action_data.get('player') == 'ai' and action_data.get('action') == 'respond':
                    state = entry['state']
                    action_str = action_data['response']
                    reward = float(entry['reward'])
                    done = bool(state.get('game_over', False))
                    next_state = game[i + 1]['state'] if i + 1 < len(game) else state

                    # Belief models the PERSON: only the person's claims (revealed when
                    # the AI responds) update it, never the AI's own claims.
                    next_belief = game_belief.copy()
                    claim = state.get('current_claim')
                    actual = state.get('current_card')
                    if claim and actual:
                        remaining = calculate_cards_remaining(state, self.config)
                        remaining[actual] = max(0, remaining[actual] - 1)  # state is pre-reveal
                        next_belief.update_after_reveal(claim, actual, remaining)

                    obs = state_to_obs(self._observable(state), self.config, game_belief)
                    next_obs = state_to_obs(self._observable(next_state), self.config, next_belief)

                    experiences.append((
                        obs,
                        self.ACTIONS.index(action_str),
                        reward,
                        next_obs,
                        done,
                    ))

                    game_belief = next_belief

        print(f"Loaded {len(experiences)} experiences")

        batch_size = 64
        if len(experiences) < batch_size:
            print(f"Need at least {batch_size} experiences — play more games first.")
            return {'games': len(games), 'experiences': len(experiences)}

        # ---- Gradient descent on SB3's network -----------------------
        steps_per_epoch = max(len(experiences) // batch_size, 10)
        total_steps = steps_per_epoch * epochs
        print(f"Training {epochs} epoch(s) × {steps_per_epoch} steps = {total_steps} gradient steps...")

        for epoch in range(epochs):
            epoch_losses = []

            for _ in range(steps_per_epoch):
                batch = random.sample(experiences, batch_size)
                obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)

                obs_t  = torch.FloatTensor(np.array(obs_b)).to(self.model.device)
                act_t  = torch.LongTensor(np.array(act_b)).unsqueeze(1).to(self.model.device)
                rew_t  = torch.FloatTensor(np.array(rew_b)).to(self.model.device)
                nobs_t = torch.FloatTensor(np.array(nobs_b)).to(self.model.device)
                done_t = torch.FloatTensor(np.array(done_b)).to(self.model.device)

                # Bellman target using frozen target network
                with torch.no_grad():
                    next_q = self.model.q_net_target(nobs_t).max(1)[0]
                    targets = rew_t + (1 - done_t) * self.model.gamma * next_q

                # Current Q-values for the taken actions
                current_q = self.model.q_net(obs_t).gather(1, act_t).squeeze(1)

                loss = F.smooth_l1_loss(current_q, targets)
                self.model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 10.0)
                self.model.policy.optimizer.step()
                epoch_losses.append(loss.item())

                # Sync target network periodically
                self._n_grad_steps += 1
                if self._n_grad_steps % self.model.target_update_interval == 0:
                    self.model.q_net_target.load_state_dict(self.model.q_net.state_dict())

            print(f"  Epoch {epoch + 1}/{epochs}: loss={np.mean(epoch_losses):.4f}")

        self.model.save(self.model_path)
        print(f"Model saved: {self.model_path}.zip")

        return {
            'games': len(games),
            'experiences': len(experiences),
            'epochs': epochs,
            'grad_steps': total_steps,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_model(self):
        """Initialise a fresh SB3 DQN (env needed only for obs/action spaces)."""
        from stable_baselines3 import DQN
        env = CockroachPokerEnv(self.config)
        return DQN(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=1e-3,
            batch_size=64,
            gamma=0.95,
            target_update_interval=100,
        )

    def _observable(self, state: dict) -> dict:
        return {
            'ai_hand': state.get('ai_hand', []),
            'player_face_up': state.get('player_face_up', []),
            'ai_face_up': state.get('ai_face_up', []),
            'current_claim': state.get('current_claim'),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_model(self):
        path = self.model_path + '.zip'
        if not os.path.exists(path):
            print(f"No SB3 model at {path}. Train first: python train.py sb3")
            return
        try:
            from stable_baselines3 import DQN
            self.model = DQN.load(self.model_path)
            print(f"SB3 DQN loaded")
        except ImportError:
            print("stable-baselines3 not installed — pip install stable-baselines3")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_statistics(self) -> dict:
        if self.model is None:
            return {'trained': False, 'grad_steps': 0}
        return {
            'trained': True,
            'grad_steps': self._n_grad_steps,
            'policy': type(self.model.policy).__name__,
            'device': str(self.model.device),
            'belief_observations': len(self.belief_state.claim_history),
        }
