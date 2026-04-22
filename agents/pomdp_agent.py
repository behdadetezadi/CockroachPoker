import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from belief import BeliefState
from core import calculate_cards_remaining, load_all_games
from agents.base import Agent


class POMDPAgent(Agent):
    """
    Tabular Q-learning agent with POMDP belief tracking.

    State space uses detailed hand/face-up counts but coarser (5-bucket)
    belief and bluff-rate features to keep the table a manageable size.
    """

    def __init__(self, config: Dict, model_path: str = 'rl_model.pkl'):
        self.config = config
        self.model_path = model_path
        self.q_table: Dict = defaultdict(lambda: defaultdict(float))

        self.learning_rate = 0.1
        self.discount_factor = 0.8
        self.epsilon = 0.2

        self.belief_state = BeliefState(config['animals'])
        self.load_model()

    # ------------------------------------------------------------------
    # Observable state helpers
    # ------------------------------------------------------------------

    def get_observable_state(self, state: Dict) -> Dict:
        """Strip hidden information — no cheating."""
        return {
            'ai_hand': state.get('ai_hand', []),
            'player_face_up': state.get('player_face_up', []),
            'ai_face_up': state.get('ai_face_up', []),
            'current_claim': state.get('current_claim'),
        }

    def _discretize(self, value: float, thresholds=(0.2, 0.4, 0.6, 0.8)) -> str:
        labels = ('very_low', 'low', 'medium', 'high', 'very_high')
        for i, t in enumerate(thresholds):
            if value < t:
                return labels[i]
        return labels[-1]

    def state_to_key(self, state: Dict, belief: Optional[BeliefState] = None) -> str:
        belief = belief or self.belief_state

        ai_counts: Dict[str, int] = defaultdict(int)
        for card in state.get('ai_hand', []):
            ai_counts[card] += 1

        p_up: Dict[str, int] = defaultdict(int)
        for card in state.get('player_face_up', []):
            p_up[card] += 1

        a_up: Dict[str, int] = defaultdict(int)
        for card in state.get('ai_face_up', []):
            a_up[card] += 1

        claim = state.get('current_claim') or 'none'
        truth_prob = belief.get_truth_probability(claim) if claim != 'none' else 0.5
        bluff_rate = belief.get_recent_bluff_rate()
        ai_has = ai_counts.get(claim, 0) if claim != 'none' else 0

        parts = [
            f"hand:{','.join(sorted(f'{k}:{v}' for k, v in ai_counts.items()))}",
            f"p_up:{','.join(sorted(f'{k}:{v}' for k, v in p_up.items()))}",
            f"a_up:{','.join(sorted(f'{k}:{v}' for k, v in a_up.items()))}",
            f"claim:{claim}",
            f"belief:{self._discretize(truth_prob)}",
            f"bluff:{self._discretize(bluff_rate)}",
            f"ai_has:{ai_has}",
        ]
        return '|'.join(parts)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def choose_response(self, state: Dict) -> str:
        obs = self.get_observable_state(state)
        key = self.state_to_key(obs)

        if random.random() < self.epsilon or not self.q_table[key]:
            return random.choice(['truth', 'bluff'])

        return max(self.q_table[key], key=self.q_table[key].get)

    def choose_card_and_claim(self, hand: List[str]) -> Tuple[Optional[str], Optional[str]]:
        if not hand:
            return None, None
        card = random.choice(hand)
        claim = card if random.random() < 0.5 else random.choice(self.config['animals'])
        return card, claim

    def update_belief(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]):
        self.belief_state.update_after_reveal(claim, actual_card, cards_remaining)

    # ------------------------------------------------------------------
    # Q-learning
    # ------------------------------------------------------------------

    def update_q_value(
        self,
        state: Dict,
        action: str,
        reward: float,
        next_state: Dict,
        is_terminal: bool = False,
        belief: Optional[BeliefState] = None,
        next_belief: Optional[BeliefState] = None,
    ):
        obs = self.get_observable_state(state)
        obs_next = self.get_observable_state(next_state)

        key = self.state_to_key(obs, belief)
        next_key = self.state_to_key(obs_next, next_belief if next_belief is not None else belief)

        current_q = self.q_table[key][action]
        max_next_q = (
            0 if is_terminal
            else max(self.q_table[next_key].values(), default=0)
        )
        self.q_table[key][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_logs(self, log_dir: str = 'game_logs') -> int:
        """Run Q-learning over all logged games. Returns number of Q-updates."""
        games = load_all_games(log_dir)
        if not games:
            return 0

        total_updates = 0

        for game in games:
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

                    self.update_q_value(
                        state, action, reward, next_state,
                        is_terminal, game_belief, next_belief,
                    )
                    total_updates += 1

                if action_data.get('action') == 'respond':
                    claim = entry['state'].get('current_claim')
                    actual = entry['state'].get('current_card')
                    if claim and actual:
                        remaining = calculate_cards_remaining(entry['state'], self.config)
                        game_belief.update_after_reveal(claim, actual, remaining)

        return total_updates

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

        metadata = {
            'q_table_size': len(self.q_table),
            'total_actions': sum(len(v) for v in self.q_table.values()),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'belief_state': self.belief_state.to_dict(),
            'timestamp': datetime.now().isoformat(),
        }
        meta_path = self.model_path.replace('.pkl', '_metadata.json')
        import json
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved: {self.model_path}  ({metadata['q_table_size']} states)")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    loaded = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), loaded)
                print(f"Model loaded: {self.model_path}  ({len(self.q_table)} states)")
            except Exception as e:
                print(f"Error loading model: {e} — starting fresh")
        else:
            print("No saved model found. Starting with empty Q-table")

    def get_statistics(self) -> Dict:
        if not self.q_table:
            q_stats = {'states': 0, 'state_action_pairs': 0,
                       'avg_q_value': 0, 'max_q_value': 0, 'min_q_value': 0}
        else:
            all_q = [v for actions in self.q_table.values() for v in actions.values()]
            q_stats = {
                'states': len(self.q_table),
                'state_action_pairs': len(all_q),
                'avg_q_value': sum(all_q) / len(all_q) if all_q else 0,
                'max_q_value': max(all_q) if all_q else 0,
                'min_q_value': min(all_q) if all_q else 0,
            }

        return {
            **q_stats,
            'belief_state': {
                'total_observations': len(self.belief_state.claim_history),
                'truth_counts': dict(self.belief_state.truth_count),
                'bluff_counts': dict(self.belief_state.bluff_count),
            },
        }
