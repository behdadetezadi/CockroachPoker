import pickle
import json
import os
import random
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class BeliefState:
    """
    Tracks beliefs about hidden information in POMDP.
    Maintains probability distributions over opponent's likely hand composition
    and behavioral patterns.
    """

    def __init__(self, animals: List[str]):
        self.animals = animals

        # Belief about opponent's hand (updated via Bayesian inference)
        self.opponent_hand_belief = {animal: 1.0 / len(animals) for animal in animals}

        # Opponent behavioral model (learned from observations)
        self.claim_history = []  # List of (claim, was_truth, response_time)
        self.truth_count = defaultdict(int)  # How often they tell truth per animal
        self.bluff_count = defaultdict(int)  # How often they bluff per animal

    def update_after_reveal(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]):
        """
        Bayesian update after card is revealed.
        Updates both hand belief and behavioral model.

        Args:
            claim: What opponent claimed
            actual_card: What the card actually was
            cards_remaining: How many of each animal are still possible
        """
        was_truth = (claim == actual_card)

        # Update behavioral model
        if was_truth:
            self.truth_count[claim] += 1
        else:
            self.bluff_count[claim] += 1

        self.claim_history.append({
            'claim': claim,
            'actual': actual_card,
            'was_truth': was_truth
        })

        # Update hand belief based on what was revealed
        # If they played a bat, they have one fewer bat
        if actual_card in self.opponent_hand_belief:
            # Reduce belief proportionally
            total_possible = sum(cards_remaining.values())
            if total_possible > 0:
                for animal in self.animals:
                    self.opponent_hand_belief[animal] = cards_remaining.get(animal, 0) / total_possible

    def get_truth_probability(self, claim: str) -> float:
        """
        Estimate probability that current claim is truth based on:
        1. Opponent's historical bluffing patterns
        2. Belief about what's in their hand

        Returns:
            Float between 0-1 representing probability claim is truthful
        """
        # Historical bluff rate for this animal
        total_claims = self.truth_count[claim] + self.bluff_count[claim]
        if total_claims > 0:
            historical_truth_rate = self.truth_count[claim] / total_claims
        else:
            historical_truth_rate = 0.5  # No data, assume 50/50

        # Belief about opponent having this animal
        hand_belief = self.opponent_hand_belief.get(claim, 0.5)

        # Combined probability (weighted average)
        # Weight history more as we get more data
        history_weight = min(total_claims / 10.0, 0.7)  # Max 70% weight on history
        belief_weight = 1.0 - history_weight

        probability = (historical_truth_rate * history_weight +
                       hand_belief * belief_weight)

        return probability

    def get_recent_bluff_rate(self, window: int = 5) -> float:
        """Get bluff rate over recent claims"""
        recent = self.claim_history[-window:] if len(self.claim_history) >= window else self.claim_history
        if not recent:
            return 0.5
        bluffs = sum(1 for h in recent if not h['was_truth'])
        return bluffs / len(recent)

    def to_dict(self) -> Dict:
        """Serialize belief state for logging"""
        return {
            'hand_belief': dict(self.opponent_hand_belief),
            'truth_counts': dict(self.truth_count),
            'bluff_counts': dict(self.bluff_count),
            'total_claims': len(self.claim_history)
        }


class POMDPAgent:
    """
    POMDP-aware Reinforcement Learning Agent using Q-Learning with Belief States.

    Key differences from naive MDP:
    1. Maintains belief state about hidden information
    2. State representation includes belief probabilities
    3. Updates beliefs via Bayesian inference
    4. Makes decisions under uncertainty
    """

    def __init__(self, config: Dict, model_path: str = 'rl_model.pkl'):
        self.config = config
        self.model_path = model_path
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.8  # Lowered - less emphasis on long-term (claims are independent)
        self.epsilon = 0.2

        # POMDP components
        self.belief_state = BeliefState(config['animals'])

        self.load_model()

    def get_observable_state(self, state: Dict) -> Dict:
        """
        Extract ONLY observable information (no cheating!).

        AI can see:
        - Its own hand
        - Face-up cards (both players)
        - Current claim

        AI CANNOT see:
        - Opponent's hand
        - Actual card being played (until revealed)

        Args:
            state: Full game state

        Returns:
            Observable state only
        """
        return {
            'ai_hand': state.get('ai_hand', []),
            'player_face_up': state.get('player_face_up', []),
            'ai_face_up': state.get('ai_face_up', []),
            'current_claim': state.get('current_claim', None),
            # Explicitly NOT including 'current_card' or 'player_hand'
        }

    def state_to_key(self, state: Dict) -> str:
        """
        Convert observable state + belief state to unique key.

        Includes:
        - Observable game state
        - Belief probability (discretized)
        - Recent behavioral patterns

        Args:
            state: Observable state dictionary

        Returns:
            String key for Q-table lookup
        """
        # AI's hand composition
        ai_hand_counts = defaultdict(int)
        for card in state.get('ai_hand', []):
            ai_hand_counts[card] += 1

        # Face-up cards
        player_faceup_counts = defaultdict(int)
        for card in state.get('player_face_up', []):
            player_faceup_counts[card] += 1

        ai_faceup_counts = defaultdict(int)
        for card in state.get('ai_face_up', []):
            ai_faceup_counts[card] += 1

        # Current claim
        claim = state.get('current_claim', 'none')

        # POMDP: Add belief probability
        if claim != 'none':
            truth_prob = self.belief_state.get_truth_probability(claim)
            belief_bucket = int(truth_prob * 10)  # Discretize to 0-10
        else:
            belief_bucket = 5  # Neutral

        # POMDP: Add recent bluff rate
        recent_bluff_rate = self.belief_state.get_recent_bluff_rate()
        bluff_bucket = int(recent_bluff_rate * 10)  # Discretize to 0-10

        # How many of claimed animal does AI have?
        ai_has_claimed = ai_hand_counts.get(claim, 0) if claim != 'none' else 0

        key_parts = [
            f"hand:{','.join(sorted([f'{k}:{v}' for k, v in ai_hand_counts.items()]))}",
            f"p_up:{','.join(sorted([f'{k}:{v}' for k, v in player_faceup_counts.items()]))}",
            f"a_up:{','.join(sorted([f'{k}:{v}' for k, v in ai_faceup_counts.items()]))}",
            f"claim:{claim}",
            f"belief:{belief_bucket}",  # POMDP component
            f"bluff_rate:{bluff_bucket}",  # POMDP component
            f"ai_has:{ai_has_claimed}"  # Useful correlation
        ]
        return '|'.join(key_parts)

    def choose_card_and_claim(self, hand: List[str]) -> Tuple[str, str]:
        """
        Choose a card to play and what to claim.
        Currently random - could be enhanced with learning.

        Args:
            hand: AI's current hand

        Returns:
            (card_to_play, claim)
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

    def choose_response(self, state: Dict) -> str:
        """
        Choose whether to call 'truth' or 'bluff' using epsilon-greedy.
        Uses ONLY observable state + belief state (no cheating!).

        Args:
            state: Full game state (will extract observable parts)

        Returns:
            'truth' or 'bluff'
        """
        # Extract only observable information
        observable_state = self.get_observable_state(state)
        state_key = self.state_to_key(observable_state)

        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon or not self.q_table[state_key]:
            return random.choice(['truth', 'bluff'])

        # Choose action with highest Q-value
        actions = self.q_table[state_key]
        return max(actions, key=actions.get)

    def update_belief(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]):
        """
        Update belief state after card is revealed.

        Args:
            claim: What was claimed
            actual_card: What card actually was
            cards_remaining: Estimated cards still in opponent's hand
        """
        self.belief_state.update_after_reveal(claim, actual_card, cards_remaining)

    def update_q_value(self, state: Dict, action: str, reward: float, next_state: Dict, is_terminal: bool = False):
        """
        Q-learning update with corrected reward structure.

        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

        Args:
            state: Observable state before action
            action: Action taken ('truth' or 'bluff')
            reward: Reward received (positive for correct, negative for wrong)
            next_state: Observable state after action
            is_terminal: Whether game ended
        """
        observable_state = self.get_observable_state(state)
        observable_next = self.get_observable_state(next_state)

        state_key = self.state_to_key(observable_state)
        next_state_key = self.state_to_key(observable_next)

        current_q = self.q_table[state_key][action]

        # Terminal states have no future value
        if is_terminal:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0

        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def train_from_logs(self, log_dir: str = 'game_logs') -> int:
        """
        Train from logged games with POMDP state reconstruction.

        Args:
            log_dir: Directory with game logs

        Returns:
            Number of updates performed
        """
        games = self._load_all_games(log_dir)

        if not games:
            return 0

        total_updates = 0

        for game in games:
            # Reset belief state for each game
            game_belief = BeliefState(self.config['animals'])

            for i, entry in enumerate(game):
                action_data = entry['action']

                # Train only on AI response actions
                if action_data.get('player') == 'ai' and action_data.get('action') == 'respond':
                    state = entry['state']
                    action = action_data['response']
                    reward = entry['reward']

                    # Reconstruct belief state at this point
                    # (In real training, we'd need to replay all prior actions)
                    # For now, use current belief approximation

                    # Check if terminal state
                    is_terminal = state.get('game_over', False)

                    # Get next state
                    next_state = game[i + 1]['state'] if i + 1 < len(game) else state

                    self.update_q_value(state, action, reward, next_state, is_terminal)
                    total_updates += 1

                # Update belief after each reveal
                if action_data.get('action') == 'respond':
                    claim = state.get('current_claim')
                    actual = state.get('current_card')
                    if claim and actual:
                        game_belief.update_after_reveal(claim, actual, {})

        return total_updates

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
        """Save Q-table and metadata to disk"""
        # Save Q-table
        with open(self.model_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

        # Save metadata including belief state stats
        metadata = {
            'q_table_size': len(self.q_table),
            'total_actions': sum(len(actions) for actions in self.q_table.values()),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'belief_state': self.belief_state.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved: {self.model_path}")
        print(f"States learned: {metadata['q_table_size']}")
        print(f"Total state-action pairs: {metadata['total_actions']}")

    def load_model(self):
        """Load Q-table from disk if exists"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    loaded_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: defaultdict(float), loaded_table)
                print(f"Model loaded from {self.model_path}")
                print(f"States in memory: {len(self.q_table)}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with empty Q-table")
        else:
            print("No saved model found. Starting with empty Q-table")

    def get_statistics(self) -> Dict:
        """Get statistics about the learned model and beliefs"""
        if not self.q_table:
            q_stats = {
                'states': 0,
                'state_action_pairs': 0,
                'avg_q_value': 0,
                'max_q_value': 0,
                'min_q_value': 0
            }
        else:
            all_q_values = []
            for state_actions in self.q_table.values():
                all_q_values.extend(state_actions.values())

            q_stats = {
                'states': len(self.q_table),
                'state_action_pairs': len(all_q_values),
                'avg_q_value': sum(all_q_values) / len(all_q_values) if all_q_values else 0,
                'max_q_value': max(all_q_values) if all_q_values else 0,
                'min_q_value': min(all_q_values) if all_q_values else 0
            }

        # Add belief statistics
        belief_stats = {
            'total_observations': len(self.belief_state.claim_history),
            'truth_counts': dict(self.belief_state.truth_count),
            'bluff_counts': dict(self.belief_state.bluff_count)
        }

        return {**q_stats, 'belief_state': belief_stats}


def train_agent(config: Dict, epochs: int = 10, log_dir: str = 'game_logs'):
    """Standalone training function"""
    print("\n" + "=" * 60)
    print("POMDP TRAINING MODE")
    print("=" * 60)

    agent = POMDPAgent(config)

    print("\nInitial Model Statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        if key != 'belief_state':
            print(f"  {key}: {value}")

    games = agent._load_all_games(log_dir)
    print(f"\nFound {len(games)} games in training data")

    if not games:
        print("No training data found! Play some games first.")
        return

    print(f"\nTraining for {epochs} epochs...")
    total_updates = 0

    for epoch in range(epochs):
        updates = agent.train_from_logs(log_dir)
        total_updates += updates
        print(f"Epoch {epoch + 1}/{epochs}: Processed {updates} updates")

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

    print("\nTraining complete!")
    print(f"Total updates performed: {total_updates}")


def main():
    """Main entry point for training/testing"""
    GAME_CONFIG = {
        'hand_size': 5,
        'animals': ['fly', 'rat', 'toad', 'bat'],
        'lose_threshold': 3,
        'cards_per_animal': 8
    }

    print("\n" + "=" * 60)
    print("COCKROACH POKER POMDP AI")
    print("=" * 60)
    print("\nOptions:")
    print("1. Train agent from existing games")
    print("2. View model statistics")
    print("3. Reset model")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        epochs = input("Enter number of training epochs (default 10): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 10
        train_agent(GAME_CONFIG, epochs=epochs)

    elif choice == '2':
        agent = POMDPAgent(GAME_CONFIG)
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
                os.remove('rl_model.pkl')
                os.remove('rl_model_metadata.json')
                print("Model deleted!")
            except FileNotFoundError:
                print("No model found.")

    elif choice == '4':
        print("Exiting...")


if __name__ == "__main__":
    main()