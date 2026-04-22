"""
Gymnasium environment for Cockroach Poker.

The agent being trained plays as the AI responder.
The opponent (player) claims randomly every turn.

Observation (19 floats for 4 animals):
  ai_hand counts (N) + player_face_up counts (N) + ai_face_up counts (N)
  + claim one-hot (N) + belief_prob (1) + bluff_rate (1) + ai_has_claimed (1)

Actions:
  0 = truth
  1 = bluff
"""

import random
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from belief import BeliefState
from core import GameState, calculate_cards_remaining, GAME_CONFIG


def _make_base():
    if _GYM_AVAILABLE:
        return gym.Env
    return object  # fallback: plain class, not a registered gym env


class CockroachPokerEnv(_make_base()):
    """
    Headless Cockroach Poker environment compatible with gymnasium.

    One step = AI responds to the current player claim, then the game
    advances (AI claims randomly, player responds randomly, player claims
    randomly) until the AI needs to respond again or the game ends.
    """

    metadata = {'render_modes': []}
    ACTIONS = ['truth', 'bluff']

    def __init__(self, config: Dict = None):
        self.config = config or GAME_CONFIG
        n = len(self.config['animals'])
        obs_shape = (n * 4 + 3,)

        if _GYM_AVAILABLE:
            super().__init__()
            self.observation_space = spaces.Box(0.0, 1.0, shape=obs_shape, dtype=np.float32)
            self.action_space = spaces.Discrete(2)

        self.state: Optional[GameState] = None
        self.belief: Optional[BeliefState] = None

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if _GYM_AVAILABLE:
            super().reset(seed=seed)

        self.state = GameState(self.config)
        self.state.deal_hands()
        self.belief = BeliefState(self.config['animals'])

        # Player claims first so the AI has something to respond to
        self._player_claim()
        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        response = self.ACTIONS[action]
        card = self.state.current_card
        claim = self.state.current_claim
        is_truth = (card == claim)

        # Apply AI response
        if (response == 'truth' and is_truth) or (response == 'bluff' and not is_truth):
            self.state.player_face_up.append(card)
            reward = 1.0
        else:
            self.state.ai_face_up.append(card)
            reward = -1.0

        # Update belief with the revealed card
        remaining = calculate_cards_remaining(self.state.to_dict(), self.config)
        self.belief.update_after_reveal(claim, card, remaining)
        self.state.current_card = None
        self.state.current_claim = None

        # Check terminal after AI response
        winner = self._check_terminal()
        if winner is not None:
            reward += 50.0 if winner == 'player' else -50.0
            return self._observe(), reward, True, False, {'winner': winner}

        # Advance: AI claims, player responds, player claims (all random)
        terminal = self._advance_game()
        if terminal:
            winner = self._check_terminal()
            if winner == 'player':
                reward += 50.0
            elif winner == 'ai':
                reward -= 50.0
            return self._observe(), reward, True, False, {'winner': winner}

        return self._observe(), reward, False, False, {}

    # ------------------------------------------------------------------
    # Internal game flow helpers
    # ------------------------------------------------------------------

    def _player_claim(self):
        """Player picks a card from their hand and makes a random claim."""
        if not self.state.player_hand:
            return
        card = random.choice(self.state.player_hand)
        self.state.player_hand.remove(card)
        claim = card if random.random() < 0.5 else random.choice(self.config['animals'])
        self.state.current_card = card
        self.state.current_claim = claim

    def _advance_game(self) -> bool:
        """
        Advance through AI-claim + player-respond + player-claim phases.
        Returns True if the game ended during this advance.
        """
        # AI claims randomly
        if self.state.ai_hand:
            card = random.choice(self.state.ai_hand)
            self.state.ai_hand.remove(card)
            claim = card if random.random() < 0.5 else random.choice(self.config['animals'])

            response = random.choice(['truth', 'bluff'])
            is_truth = (card == claim)
            if (response == 'truth' and is_truth) or (response == 'bluff' and not is_truth):
                self.state.ai_face_up.append(card)
            else:
                self.state.player_face_up.append(card)

            remaining = calculate_cards_remaining(self.state.to_dict(), self.config)
            self.belief.update_after_reveal(claim, card, remaining)

            if self._check_terminal() is not None:
                return True

        # Player claims to set up the AI's next response
        if self.state.player_hand:
            self._player_claim()
            return False

        # No more cards — game over
        return True

    def _check_terminal(self) -> Optional[str]:
        winner = self.state.check_loss_condition()
        if not winner:
            winner = self.state.check_empty_hands()
        if winner:
            self.state.game_over = True
            self.state.winner = winner
        return winner

    def _observe(self) -> np.ndarray:
        max_cards = self.config['cards_per_animal']
        features: list = []

        ai_counts: Dict[str, int] = defaultdict(int)
        for card in self.state.ai_hand:
            ai_counts[card] += 1

        p_up: Dict[str, int] = defaultdict(int)
        for card in self.state.player_face_up:
            p_up[card] += 1

        a_up: Dict[str, int] = defaultdict(int)
        for card in self.state.ai_face_up:
            a_up[card] += 1

        for animal in self.config['animals']:
            features.append(ai_counts[animal] / max_cards)
        for animal in self.config['animals']:
            features.append(p_up[animal] / max_cards)
        for animal in self.config['animals']:
            features.append(a_up[animal] / max_cards)

        claim = self.state.current_claim
        for animal in self.config['animals']:
            features.append(1.0 if claim == animal else 0.0)

        if claim:
            features.append(self.belief.get_truth_probability(claim))
            features.append(self.belief.get_recent_bluff_rate())
            features.append(ai_counts.get(claim, 0) / max_cards)
        else:
            features += [0.5, 0.5, 0.0]

        return np.array(features, dtype=np.float32)
