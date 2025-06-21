"""
Test cases for game logic and models
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game.models import Card, Player, GameState, PendingChallenge, PlayerType, GamePhase
from src.game.game_manager import GameManager
from src.config import CREATURES


class TestCard:
    """Test Card model"""

    def test_card_creation(self):
        card = Card("test-id", "cockroach")
        assert card.id == "test-id"
        assert card.creature == "cockroach"

    def test_card_auto_id(self):
        card = Card("", "rat")
        assert card.id != ""
        assert len(card.id) > 0


class TestPlayer:
    """Test Player model"""

    def test_player_creation(self):
        player = Player(PlayerType.HUMAN)
        assert player.player_type == PlayerType.HUMAN
        assert len(player.hand) == 0
        assert len(player.cards_on_table) == 0
        assert player.score == 0

    def test_has_lost_condition(self):
        player = Player(PlayerType.HUMAN)

        # Add 4 of the same creature
        for i in range(4):
            card = Card(f"cockroach-{i}", "cockroach")
            player.cards_on_table.append(card)

        assert player.has_lost() == True

    def test_has_won_condition(self):
        player = Player(PlayerType.HUMAN)
        assert player.has_won() == True  # Empty hand = win

        # Add a card to hand
        player.hand.append(Card("test", "rat"))
        assert player.has_won() == False


class TestGameState:
    """Test GameState model"""

    def test_game_state_creation(self):
        state = GameState()
        assert state.current_player == PlayerType.HUMAN
        assert state.phase == GamePhase.SETUP
        assert state.round_number == 1
        assert state.pending_challenge is None

    def test_switch_current_player(self):
        state = GameState()
        assert state.current_player == PlayerType.HUMAN

        state.switch_current_player()
        assert state.current_player == PlayerType.AI

        state.switch_current_player()
        assert state.current_player == PlayerType.HUMAN

    def test_get_current_player(self):
        state = GameState()
        current = state.get_current_player()
        assert current == state.human_player

        state.current_player = PlayerType.AI
        current = state.get_current_player()
        assert current == state.ai_player

    def test_get_opponent(self):
        state = GameState()
        opponent = state.get_opponent(PlayerType.HUMAN)
        assert opponent == state.ai_player

        opponent = state.get_opponent(PlayerType.AI)
        assert opponent == state.human_player


class TestGameManager:
    """Test GameManager functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.game_manager = GameManager()

    def teardown_method(self):
        """Cleanup after each test"""
        self.game_manager.cleanup()

    def test_game_manager_creation(self):
        assert self.game_manager.game_state is not None
        assert self.game_manager.ai_strategy is not None
        assert self.game_manager.analytics is not None

    def test_set_ai_strategy(self):
        self.game_manager.set_ai_strategy('random')
        assert 'Random' in self.game_manager.ai_strategy.name

        self.game_manager.set_ai_strategy('pattern')
        assert 'Pattern' in self.game_manager.ai_strategy.name

    def test_start_new_game(self):
        self.game_manager.start_new_game()

        state = self.game_manager.game_state
        assert state.phase == GamePhase.PLAYING
        assert len(state.human_player.hand) == 8
        assert len(state.ai_player.hand) == 8
        assert state.current_player == PlayerType.HUMAN

    def test_human_pass_card(self):
        self.game_manager.start_new_game()
        state = self.game_manager.game_state

        # Get a card from human hand
        card = state.human_player.hand[0]
        claimed_creature = "spider"

        # Pass the card
        success = self.game_manager.human_pass_card(card, claimed_creature)
        assert success == True

        # Check game state
        assert state.phase == GamePhase.CHALLENGE
        assert state.pending_challenge is not None
        assert state.pending_challenge.card == card
        assert state.pending_challenge.claimed_creature == claimed_creature
        assert card not in state.human_player.hand

    def test_invalid_card_pass(self):
        self.game_manager.start_new_game()
        state = self.game_manager.game_state

        # Try to pass a card not in hand
        fake_card = Card("fake", "cockroach")
        success = self.game_manager.human_pass_card(fake_card, "rat")
        assert success == False
        assert state.phase == GamePhase.PLAYING  # Should remain unchanged

    def test_challenge_resolution(self):
        self.game_manager.start_new_game()
        state = self.game_manager.game_state

        # Setup a challenge
        card = Card("test", "cockroach")
        state.human_player.hand.append(card)
        self.game_manager.human_pass_card(card, "rat")  # Bluff

        # Human challenges (should be correct since it's a bluff)
        success = self.game_manager.human_challenge_decision(True)
        assert success == True

        # Card should go to human (challenger was correct)
        assert card in state.human_player.cards_on_table

    def test_analytics_collection(self):
        self.game_manager.start_new_game()

        initial_decisions = self.game_manager.analytics.total_decisions

        # Make a decision
        state = self.game_manager.game_state
        card = state.human_player.hand[0]
        self.game_manager.human_pass_card(card, "spider")
        self.game_manager.human_challenge_decision(False)

        # Check analytics updated
        assert self.game_manager.analytics.total_decisions > initial_decisions


class TestDeckCreation:
    """Test deck creation and card distribution"""

    def test_deck_has_correct_cards(self):
        game_manager = GameManager()
        deck = game_manager._create_deck()

        # Check total cards
        assert len(deck) == len(CREATURES) * 8

        # Check each creature appears 8 times
        creature_counts = {}
        for card in deck:
            creature_counts[card.creature] = creature_counts.get(card.creature, 0) + 1

        for creature in CREATURES:
            assert creature_counts[creature] == 8

        game_manager.cleanup()


class TestWinConditions:
    """Test various win conditions"""

    def setup_method(self):
        self.game_manager = GameManager()
        self.game_manager.start_new_game()

    def teardown_method(self):
        self.game_manager.cleanup()

    def test_human_loses_with_four_same_creatures(self):
        state = self.game_manager.game_state

        # Give human 4 cockroaches
        for i in range(4):
            card = Card(f"cockroach-{i}", "cockroach")
            state.human_player.cards_on_table.append(card)

        # Check win condition
        assert state.human_player.has_lost() == True
        assert state.ai_player.has_lost() == False

    def test_human_wins_with_empty_hand(self):
        state = self.game_manager.game_state

        # Empty human hand
        state.human_player.hand.clear()

        # Check win condition
        assert state.human_player.has_won() == True
        assert state.ai_player.has_won() == False


if __name__ == "__main__":
    pytest.main([__file__])