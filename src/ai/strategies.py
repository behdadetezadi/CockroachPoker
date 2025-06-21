"""
AI Strategy implementations for bluff detection
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json

from ..game.models import GameState, DecisionData, FacialExpression, PlayerType
from ..config import CREATURES, MAX_SAME_CREATURE


class AIStrategy(ABC):
    """Base class for AI strategies"""

    def __init__(self, name: str):
        self.name = name
        self.game_history: List[Dict[str, Any]] = []
        self.player_patterns: Dict[str, Any] = defaultdict(list)
        self.learning_data: Dict[str, Any] = {}

    @abstractmethod
    def make_challenge_decision(self, game_state: GameState,
                                challenge_data: Dict[str, Any]) -> bool:
        """Decide whether to challenge a claim"""
        pass

    @abstractmethod
    def make_bluff_decision(self, game_state: GameState,
                            card_to_pass) -> str:
        """Decide what to claim when passing a card"""
        pass

    def update_history(self, game_state: GameState, decision: Dict[str, Any],
                       outcome: Dict[str, Any]):
        """Update the strategy's learning history"""
        history_entry = {
            'game_state': self._serialize_game_state(game_state),
            'decision': decision,
            'outcome': outcome,
            'timestamp': time.time()
        }
        self.game_history.append(history_entry)
        self._update_patterns(decision, outcome)

    def _serialize_game_state(self, game_state: GameState) -> Dict[str, Any]:
        """Serialize game state for storage"""
        return {
            'round': game_state.round_number,
            'current_player': game_state.current_player.value,
            'human_hand_size': len(game_state.human_player.hand),
            'ai_hand_size': len(game_state.ai_player.hand),
            'human_cards_count': len(game_state.human_player.cards_on_table),
            'ai_cards_count': len(game_state.ai_player.cards_on_table)
        }

    def _update_patterns(self, decision: Dict[str, Any], outcome: Dict[str, Any]):
        """Update player behavior patterns"""
        if 'player_type' in decision:
            player_key = decision['player_type']
            self.player_patterns[player_key].append({
                'decision': decision,
                'outcome': outcome,
                'timestamp': time.time()
            })


class RandomStrategy(AIStrategy):
    """Random decision making strategy (baseline)"""

    def __init__(self):
        super().__init__("Random AI")

    def make_challenge_decision(self, game_state: GameState,
                                challenge_data: Dict[str, Any]) -> bool:
        """Make random challenge decisions"""
        return random.random() > 0.5

    def make_bluff_decision(self, game_state: GameState, card_to_pass) -> str:
        """Make random bluff decisions"""
        should_bluff = random.random() > 0.5
        if should_bluff:
            available_creatures = [c for c in CREATURES if c != card_to_pass.creature]
            return random.choice(available_creatures)
        return card_to_pass.creature


class PatternBasedStrategy(AIStrategy):
    """Pattern-based bluff detection using behavioral analysis"""

    def __init__(self):
        super().__init__("Pattern-Based AI")
        self.bluff_indicators = {
            'decision_time_threshold': 3.0,  # seconds
            'quick_decision_threshold': 1.0,
            'confidence_threshold': 0.7,
            'nervousness_weight': 0.4,
            'pattern_weight': 0.3,
            'timing_weight': 0.3
        }

    def make_challenge_decision(self, game_state: GameState,
                                challenge_data: Dict[str, Any]) -> bool:
        """Make challenge decision based on patterns"""
        suspicion_score = self._calculate_suspicion_score(game_state, challenge_data)
        return suspicion_score > 0.5

    def _calculate_suspicion_score(self, game_state: GameState,
                                   challenge_data: Dict[str, Any]) -> float:
        """Calculate suspicion score based on multiple factors"""
        score = 0.0

        # Analyze decision timing
        decision_time = challenge_data.get('decision_time', 0)
        if decision_time > self.bluff_indicators['decision_time_threshold']:
            score += 0.3  # Long thinking is suspicious
        elif decision_time < self.bluff_indicators['quick_decision_threshold']:
            score += 0.2  # Too quick is also suspicious

        # Analyze facial expression
        facial_expr = challenge_data.get('facial_expression')
        if facial_expr:
            if facial_expr.confidence < self.bluff_indicators['confidence_threshold']:
                score += 0.2
            if facial_expr.emotion == 'nervous':
                score += self.bluff_indicators['nervousness_weight']

        # Analyze historical patterns
        player_history = [h for h in self.game_history
                          if h['game_state']['current_player'] == 'human']
        if len(player_history) > 0:
            bluff_rate = sum(1 for h in player_history
                             if h['decision'].get('was_bluff', False)) / len(player_history)
            score += bluff_rate * self.bluff_indicators['pattern_weight']

        # Analyze game state context
        claimed_creature = challenge_data.get('claimed_creature', '')
        human_cards = game_state.human_player.cards_on_table
        human_creature_counts = defaultdict(int)
        for card in human_cards:
            human_creature_counts[card.creature] += 1

        # If human already has many of the claimed creature, they're less likely to want more
        if human_creature_counts[claimed_creature] >= 2:
            score += 0.3

        return min(score, 1.0)  # Cap at 1.0

    def make_bluff_decision(self, game_state: GameState, card_to_pass) -> str:
        """Strategic bluffing based on game analysis"""
        # Analyze opponent's situation
        human_cards = game_state.human_player.cards_on_table
        human_creature_counts = defaultdict(int)
        for card in human_cards:
            human_creature_counts[card.creature] += 1

        # Find creatures the human has many of (dangerous for them)
        dangerous_creatures = [creature for creature, count in human_creature_counts.items()
                               if count >= 2]

        # Strategic bluffing: claim a creature the human doesn't want
        if dangerous_creatures and random.random() > 0.3:
            return random.choice(dangerous_creatures)

        # Otherwise, mostly tell the truth
        should_bluff = random.random() > 0.7
        if should_bluff:
            available_creatures = [c for c in CREATURES if c != card_to_pass.creature]
            return random.choice(available_creatures)

        return card_to_pass.creature


class ReinforcementLearningStrategy(AIStrategy):
    """Q-learning based strategy for bluff detection"""

    def __init__(self):
        super().__init__("RL Agent")
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: {'challenge': 0.0, 'accept': 0.0})
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.state_features = [
            'timing_category', 'expression_confidence', 'game_phase',
            'opponent_pattern', 'creature_context'
        ]

    def _get_state_key(self, game_state: GameState, challenge_data: Dict[str, Any]) -> str:
        """Generate state key for Q-table"""
        decision_time = challenge_data.get('decision_time', 0)
        timing_cat = 'slow' if decision_time > 3.0 else 'fast' if decision_time < 1.0 else 'normal'

        facial_expr = challenge_data.get('facial_expression')
        expr_conf = 'high' if facial_expr and facial_expr.confidence > 0.7 else 'low'

        game_phase = 'early' if game_state.round_number < 5 else 'late'

        # Opponent pattern analysis
        recent_history = self.game_history[-5:] if len(self.game_history) >= 5 else self.game_history
        bluff_rate = 'high' if recent_history and \
                               sum(1 for h in recent_history if h['decision'].get('was_bluff', False)) / len(
            recent_history) > 0.5 \
            else 'low'

        # Creature context
        claimed_creature = challenge_data.get('claimed_creature', '')
        human_cards = game_state.human_player.cards_on_table
        creature_count = sum(1 for card in human_cards if card.creature == claimed_creature)
        creature_context = 'many' if creature_count >= 2 else 'few'

        state_key = f"{timing_cat}_{expr_conf}_{game_phase}_{bluff_rate}_{creature_context}"
        return state_key

    def make_challenge_decision(self, game_state: GameState,
                                challenge_data: Dict[str, Any]) -> bool:
        """Make decision using epsilon-greedy Q-learning"""
        state_key = self._get_state_key(game_state, challenge_data)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.random() > 0.5

        # Exploit: choose action with highest Q-value
        q_values = self.q_table[state_key]
        return q_values['challenge'] > q_values['accept']

    def update_q_value(self, state_key: str, action: str, reward: float,
                       next_state_key: str):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state_key][action]
        next_q_values = self.q_table[next_state_key]
        max_next_q = max(next_q_values['challenge'], next_q_values['accept'])

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def update_history(self, game_state: GameState, decision: Dict[str, Any],
                       outcome: Dict[str, Any]):
        """Update history and Q-values"""
        super().update_history(game_state, decision, outcome)

        # Update Q-values if this was a challenge decision
        if 'challenge_decision' in decision:
            state_key = decision.get('state_key', '')
            action = 'challenge' if decision['challenge_decision'] else 'accept'
            reward = 1.0 if outcome.get('correct', False) else -1.0
            next_state_key = outcome.get('next_state_key', state_key)

            self.update_q_value(state_key, action, reward, next_state_key)

    def make_bluff_decision(self, game_state: GameState, card_to_pass) -> str:
        """Simple bluffing strategy for RL agent"""
        # For now, use simple strategy - can be enhanced with separate Q-table
        should_bluff = random.random() > 0.6
        if should_bluff:
            available_creatures = [c for c in CREATURES if c != card_to_pass.creature]
            return random.choice(available_creatures)
        return card_to_pass.creature


class NeuralNetworkStrategy(AIStrategy):
    """Neural network based strategy (simplified implementation)"""

    def __init__(self):
        super().__init__("Neural Network AI")
        # This would typically use a real neural network library
        # For now, we'll simulate with weighted features
        self.weights = {
            'decision_time': random.uniform(-1, 1),
            'expression_confidence': random.uniform(-1, 1),
            'expression_nervousness': random.uniform(-1, 1),
            'historical_bluff_rate': random.uniform(-1, 1),
            'creature_count': random.uniform(-1, 1),
            'game_phase': random.uniform(-1, 1)
        }
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01

    def _extract_features(self, game_state: GameState,
                          challenge_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features for neural network"""
        features = {}

        # Timing features
        decision_time = challenge_data.get('decision_time', 0)
        features['decision_time'] = min(decision_time / 5.0, 1.0)  # Normalize

        # Expression features
        facial_expr = challenge_data.get('facial_expression')
        if facial_expr:
            features['expression_confidence'] = facial_expr.confidence
            features['expression_nervousness'] = 1.0 if facial_expr.emotion == 'nervous' else 0.0
        else:
            features['expression_confidence'] = 0.5
            features['expression_nervousness'] = 0.0

        # Historical pattern features
        player_history = [h for h in self.game_history
                          if h['game_state']['current_player'] == 'human']
        if len(player_history) > 0:
            bluff_rate = sum(1 for h in player_history
                             if h['decision'].get('was_bluff', False)) / len(player_history)
            features['historical_bluff_rate'] = bluff_rate
        else:
            features['historical_bluff_rate'] = 0.5

        # Game context features
        claimed_creature = challenge_data.get('claimed_creature', '')
        human_cards = game_state.human_player.cards_on_table
        creature_count = sum(1 for card in human_cards if card.creature == claimed_creature)
        features['creature_count'] = min(creature_count / MAX_SAME_CREATURE, 1.0)

        # Game phase
        features['game_phase'] = min(game_state.round_number / 20.0, 1.0)

        return features

    def _forward_pass(self, features: Dict[str, float]) -> float:
        """Simple forward pass through network"""
        output = self.bias
        for feature_name, value in features.items():
            if feature_name in self.weights:
                output += self.weights[feature_name] * value

        # Sigmoid activation
        return 1.0 / (1.0 + pow(2.718, -output))

    def make_challenge_decision(self, game_state: GameState,
                                challenge_data: Dict[str, Any]) -> bool:
        """Make decision using neural network"""
        features = self._extract_features(game_state, challenge_data)
        output = self._forward_pass(features)
        return output > 0.5

    def update_weights(self, features: Dict[str, float], target: float, prediction: float):
        """Update network weights using gradient descent"""
        error = target - prediction

        # Update weights
        for feature_name, value in features.items():
            if feature_name in self.weights:
                self.weights[feature_name] += self.learning_rate * error * value

        # Update bias
        self.bias += self.learning_rate * error

    def update_history(self, game_state: GameState, decision: Dict[str, Any],
                       outcome: Dict[str, Any]):
        """Update history and train network"""
        super().update_history(game_state, decision, outcome)

        # Train network if this was a challenge decision
        if 'challenge_decision' in decision and 'features' in decision:
            target = 1.0 if outcome.get('correct', False) else 0.0
            prediction = decision.get('prediction', 0.5)
            features = decision['features']

            self.update_weights(features, target, prediction)

    def make_bluff_decision(self, game_state: GameState, card_to_pass) -> str:
        """Strategic bluffing for neural network"""
        # Use strategic approach similar to pattern-based
        should_bluff = random.random() > 0.65
        if should_bluff:
            available_creatures = [c for c in CREATURES if c != card_to_pass.creature]
            return random.choice(available_creatures)
        return card_to_pass.creature


# Strategy factory
def create_strategy(strategy_type: str) -> AIStrategy:
    """Factory function to create AI strategies"""
    strategies = {
        'random': RandomStrategy,
        'pattern': PatternBasedStrategy,
        'reinforcement': ReinforcementLearningStrategy,
        'neural': NeuralNetworkStrategy
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategies[strategy_type]()