"""
Test cases for AI strategies
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai.strategies import (
    RandomStrategy, PatternBasedStrategy, ReinforcementLearningStrategy,
    NeuralNetworkStrategy, create_strategy
)
from src.game.models import GameState, Card, Player, PlayerType
from src.detection.face_detection import FacialExpression


class TestAIStrategyFactory:
    """Test strategy creation factory"""

    def test_create_random_strategy(self):
        strategy = create_strategy('random')
        assert isinstance(strategy, RandomStrategy)
        assert strategy.name == "Random AI"

    def test_create_pattern_strategy(self):
        strategy = create_strategy('pattern')
        assert isinstance(strategy, PatternBasedStrategy)
        assert strategy.name == "Pattern-Based AI"

    def test_create_rl_strategy(self):
        strategy = create_strategy('reinforcement')
        assert isinstance(strategy, ReinforcementLearningStrategy)
        assert strategy.name == "RL Agent"

    def test_create_neural_strategy(self):
        strategy = create_strategy('neural')
        assert isinstance(strategy, NeuralNetworkStrategy)
        assert strategy.name == "Neural Network AI"

    def test_invalid_strategy_type(self):
        with pytest.raises(ValueError):
            create_strategy('invalid_strategy')


class TestRandomStrategy:
    """Test Random AI strategy"""

    def setup_method(self):
        self.strategy = RandomStrategy()
        self.game_state = GameState()

    def test_strategy_creation(self):
        assert self.strategy.name == "Random AI"
        assert len(self.strategy.game_history) == 0

    def test_make_challenge_decision(self):
        challenge_data = {
            'claimed_creature': 'cockroach',
            'decision_time': 2.0,
            'facial_expression': FacialExpression('neutral', 0.8)
        }

        # Test multiple decisions to ensure randomness
        decisions = []
        for _ in range(10):
            decision = self.strategy.make_challenge_decision(self.game_state, challenge_data)
            assert isinstance(decision, bool)
            decisions.append(decision)

        # Should have some variation (not all same)
        assert len(set(decisions)) > 1

    def test_make_bluff_decision(self):
        card = Card("test", "cockroach")

        # Test multiple bluff decisions
        claims = []
        for _ in range(10):
            claim = self.strategy.make_bluff_decision(self.game_state, card)
            assert isinstance(claim, str)
            claims.append(claim)

        # Should include both truth and bluffs
        assert 'cockroach' in claims  # Truth telling
        # Should have some variation
        assert len(set(claims)) >= 1


class TestPatternBasedStrategy:
    """Test Pattern-Based AI strategy"""

    def setup_method(self):
        self.strategy = PatternBasedStrategy()
        self.game_state = GameState()

    def test_suspicion_score_calculation(self):
        # Test with suspicious timing (too slow)
        challenge_data = {
            'claimed_creature': 'spider',
            'decision_time': 4.0,  # Slow decision
            'facial_expression': FacialExpression('nervous', 0.6)
        }

        score = self.strategy._calculate_suspicion_score(self.game_state, challenge_data)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be suspicious

    def test_confident_expression_low_suspicion(self):
        # Test with confident expression
        challenge_data = {
            'claimed_creature': 'rat',
            'decision_time': 2.0,  # Normal timing
            'facial_expression': FacialExpression('confident', 0.9)
        }

        score = self.strategy._calculate_suspicion_score(self.game_state, challenge_data)
        assert score < 0.5  # Should be less suspicious

    def test_strategic_bluffing(self):
        # Setup game state where human has many cockroaches
        for i in range(3):
            card = Card(f"cockroach-{i}", "cockroach")
            self.game_state.human_player.cards_on_table.append(card)

        test_card = Card("test", "spider")

        # Test bluff decision - should sometimes claim cockroach to hurt human
        claims = []
        for _ in range(20):
            claim = self.strategy.make_bluff_decision(self.game_state, test_card)
            claims.append(claim)

        # Should sometimes claim cockroach strategically
        assert 'cockroach' in claims


class TestReinforcementLearningStrategy:
    """Test RL Agent strategy"""

    def setup_method(self):
        self.strategy = ReinforcementLearningStrategy()
        self.game_state = GameState()

    def test_state_key_generation(self):
        challenge_data = {
            'decision_time': 2.5,
            'facial_expression': FacialExpression('thinking', 0.7),
            'claimed_creature': 'fly'
        }

        state_key = self.strategy._get_state_key(self.game_state, challenge_data)
        assert isinstance(state_key, str)
        assert len(state_key) > 0

        # Should be consistent for same inputs
        state_key2 = self.strategy._get_state_key(self.game_state, challenge_data)
        assert state_key == state_key2

    def test_q_value_updates(self):
        state_key = "test_state"
        action = "challenge"
        reward = 1.0
        next_state_key = "next_state"

        # Initialize Q-values
        initial_q = self.strategy.q_table[state_key][action]

        # Update Q-value
        self.strategy.update_q_value(state_key, action, reward, next_state_key)

        # Q-value should change
        updated_q = self.strategy.q_table[state_key][action]
        assert updated_q != initial_q

    def test_exploration_vs_exploitation(self):
        # Set high epsilon for exploration
        self.strategy.epsilon = 0.9

        challenge_data = {
            'decision_time': 2.0,
            'facial_expression': FacialExpression('neutral', 0.8),
            'claimed_creature': 'bat'
        }

        # Test multiple decisions - should be random due to high epsilon
        decisions = []
        for _ in range(10):
            decision = self.strategy.make_challenge_decision(self.game_state, challenge_data)
            decisions.append(decision)

        # Should have variation due to exploration
        assert len(set(decisions)) > 1


class TestNeuralNetworkStrategy:
    """Test Neural Network AI strategy"""

    def setup_method(self):
        self.strategy = NeuralNetworkStrategy()
        self.game_state = GameState()

    def test_feature_extraction(self):
        challenge_data = {
            'decision_time': 3.5,
            'facial_expression': FacialExpression('nervous', 0.6),
            'claimed_creature': 'scorpion'
        }

        features = self.strategy._extract_features(self.game_state, challenge_data)

        # Check required features exist
        required_features = [
            'decision_time', 'expression_confidence', 'expression_nervousness',
            'historical_bluff_rate', 'creature_count', 'game_phase'
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0.0 <= features[feature] <= 1.0

    def test_forward_pass(self):
        features = {
            'decision_time': 0.5,
            'expression_confidence': 0.8,
            'expression_nervousness': 1.0,
            'historical_bluff_rate': 0.6,
            'creature_count': 0.3,
            'game_phase': 0.2
        }

        output = self.strategy._forward_pass(features)
        assert isinstance(output, float)
        assert 0.0 <= output <= 1.0  # Sigmoid output

    def test_weight_updates(self):
        features = {
            'decision_time': 0.5,
            'expression_confidence': 0.8
        }

        # Store initial weights
        initial_weights = self.strategy.weights.copy()
        initial_bias = self.strategy.bias

        # Update weights
        target = 1.0
        prediction = 0.3
        self.strategy.update_weights(features, target, prediction)

        # Weights should change
        assert self.strategy.weights != initial_weights
        assert self.strategy.bias != initial_bias


class TestAIStrategyComparison:
    """Compare different AI strategies"""

    def setup_method(self):
        self.strategies = [
            RandomStrategy(),
            PatternBasedStrategy(),
            ReinforcementLearningStrategy(),
            NeuralNetworkStrategy()
        ]
        self.game_state = GameState()

    def test_all_strategies_make_decisions(self):
        challenge_data = {
            'claimed_creature': 'toad',
            'decision_time': 2.0,
            'facial_expression': FacialExpression('thinking', 0.7)
        }

        for strategy in self.strategies:
            decision = strategy.make_challenge_decision(self.game_state, challenge_data)
            assert isinstance(decision, bool)

            card = Card("test", "fly")
            bluff_claim = strategy.make_bluff_decision(self.game_state, card)
            assert isinstance(bluff_claim, str)
            assert bluff_claim in ['cockroach', 'rat', 'stinkbug', 'fly', 'spider', 'scorpion', 'toad', 'bat']

    def test_strategy_learning_updates(self):
        # Test that strategies can update their history
        decision = {'test_decision': True}
        outcome = {'test_outcome': True}

        for strategy in self.strategies:
            initial_history_length = len(strategy.game_history)
            strategy.update_history(self.game_state, decision, outcome)
            assert len(strategy.game_history) > initial_history_length


if __name__ == "__main__":
    pytest.main([__file__])