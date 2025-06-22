"""
Game Manager - Core game logic and state management
"""

import random
import time
import threading
from typing import Optional, List, Callable, Dict, Any

from .models import (
    Card, Player, GameState, PendingChallenge, DecisionData,
    GamePhase, PlayerType, GameAnalytics
)
from ..ai.strategies import AIStrategy, create_strategy
from ..detection.face_detection import FaceDetector, EmotionAnalyzer, FaceDetectionConfig
from ..config import CREATURES, CARDS_PER_CREATURE, HAND_SIZE, MAX_SAME_CREATURE


class GameManager:
    """Manages game state, AI decisions, and data collection"""

    def __init__(self):
        # Game state
        self.game_state = GameState()
        self.analytics = GameAnalytics()

        # AI and detection
        self.ai_strategy: Optional[AIStrategy] = None
        self.face_detector = FaceDetector(FaceDetectionConfig())
        self.emotion_analyzer = EmotionAnalyzer()

        # Timing and decisions
        self.decision_start_time: Optional[float] = None
        self.ai_decision_delay = 1.5  # seconds
        self.pending_ai_action: Optional[threading.Timer] = None

        # Callbacks for UI updates
        self.state_change_callback: Optional[Callable] = None
        self.message_callback: Optional[Callable[[str], None]] = None

        # Initialize with pattern-based AI
        self.set_ai_strategy('pattern')

    def set_state_change_callback(self, callback: Callable):
        """Set callback for when game state changes"""
        self.state_change_callback = callback

    def set_message_callback(self, callback: Callable[[str], None]):
        """Set callback for displaying messages"""
        self.message_callback = callback

    def _notify_state_change(self):
        """Notify UI of state change"""
        if self.state_change_callback:
            self.state_change_callback()

    def _show_message(self, message: str):
        """Show message to user"""
        self.game_state.last_action_message = message
        if self.message_callback:
            self.message_callback(message)
        self._notify_state_change()

    def set_ai_strategy(self, strategy_type: str):
        """Set the AI strategy type"""
        try:
            self.ai_strategy = create_strategy(strategy_type)
            self._show_message(f"AI strategy changed to: {self.ai_strategy.name}")
        except ValueError as e:
            self._show_message(f"Error setting AI strategy: {e}")

    def start_new_game(self):
        """Initialize a new game"""
        # Create and shuffle deck
        deck = self._create_deck()
        random.shuffle(deck)

        # Deal cards
        human_hand = deck[:HAND_SIZE]
        ai_hand = deck[HAND_SIZE:HAND_SIZE * 2]

        # Reset game state
        self.game_state = GameState(
            human_player=Player(PlayerType.HUMAN, hand=human_hand),
            ai_player=Player(PlayerType.AI, hand=ai_hand),
            current_player=PlayerType.HUMAN,
            phase=GamePhase.PLAYING,
            round_number=1
        )

        # Start face detection
        self.face_detector.start_detection(self._on_face_detection_update)

        # Reset analytics for new game
        self.emotion_analyzer.reset_patterns()

        self._show_message("New game started! Select a card to pass to the AI.")
        self._notify_state_change()

    def _create_deck(self) -> List[Card]:
        """Create a shuffled deck of cards"""
        deck = []
        for creature in CREATURES:
            for i in range(CARDS_PER_CREATURE):
                deck.append(Card(f"{creature}-{i}", creature))
        return deck

    def _on_face_detection_update(self, expression):
        """Handle face detection updates"""
        self.emotion_analyzer.add_expression(expression)

    def get_current_facial_expression(self):
        """Get current facial expression data"""
        return self.face_detector.get_current_expression()

    def get_emotion_analysis(self):
        """Get current emotion analysis"""
        return self.emotion_analyzer.get_analysis_summary()

    def human_pass_card(self, card: Card, claimed_creature: str) -> bool:
        """Human player passes a card to AI"""
        if (self.game_state.phase != GamePhase.PLAYING or
                self.game_state.current_player != PlayerType.HUMAN or
                self.game_state.pending_challenge is not None):
            return False

        # Validate card is in human's hand
        if card not in self.game_state.human_player.hand:
            return False

        # Remove card from human hand
        self.game_state.human_player.hand.remove(card)

        # Create challenge
        is_bluff = claimed_creature != card.creature
        challenge = PendingChallenge(
            card=card,
            claimed_creature=claimed_creature,
            is_bluff=is_bluff,
            passer=PlayerType.HUMAN,
            receiver=PlayerType.AI
        )

        self.game_state.pending_challenge = challenge
        self.game_state.phase = GamePhase.CHALLENGE

        self._show_message(f'You passed a "{claimed_creature}" to the AI. AI is deciding...')

        # Schedule AI decision
        self._schedule_ai_challenge_decision()

        return True

    def _schedule_ai_challenge_decision(self):
        """Schedule AI to make challenge decision after delay"""

        def make_ai_decision():
            if self.game_state.pending_challenge:
                self._handle_ai_challenge_decision()

        self.pending_ai_action = threading.Timer(self.ai_decision_delay, make_ai_decision)
        self.pending_ai_action.start()

    def _handle_ai_challenge_decision(self):
        """AI decides whether to challenge human's claim"""
        if not self.game_state.pending_challenge or not self.ai_strategy:
            return

        challenge = self.game_state.pending_challenge

        # Prepare challenge data for AI
        challenge_data = {
            'claimed_creature': challenge.claimed_creature,
            'decision_time': random.uniform(1.0, 4.0),  # Mock AI decision time
            'facial_expression': self.face_detector.get_current_expression(),
            'emotion_analysis': self.emotion_analyzer.get_analysis_summary()
        }

        # AI makes decision
        ai_challenges = self.ai_strategy.make_challenge_decision(
            self.game_state, challenge_data
        )

        # Record decision data
        decision_data = DecisionData(
            timestamp=time.time(),
            claimed_creature=challenge.claimed_creature,
            actual_creature=challenge.card.creature,
            was_bluff=challenge.is_bluff,
            player_challenged=ai_challenges,
            decision_time=challenge_data['decision_time'],
            facial_expression=challenge_data['facial_expression'],
            game_round=self.game_state.round_number,
            confidence_score=challenge_data['emotion_analysis'].get('deception_likelihood', 0.0)
        )

        # Resolve challenge
        self._resolve_challenge(ai_challenges, decision_data)

    def human_challenge_decision(self, challenges: bool):
        """Human decides whether to challenge AI's claim"""
        if (not self.game_state.pending_challenge or
                self.game_state.pending_challenge.receiver != PlayerType.HUMAN):
            return False

        # Calculate decision time
        decision_time = 0.0
        if self.decision_start_time:
            decision_time = time.time() - self.decision_start_time

        challenge = self.game_state.pending_challenge

        # Record decision data
        decision_data = DecisionData(
            timestamp=time.time(),
            claimed_creature=challenge.claimed_creature,
            actual_creature=challenge.card.creature,
            was_bluff=challenge.is_bluff,
            player_challenged=challenges,
            decision_time=decision_time,
            facial_expression=self.face_detector.get_current_expression(),
            game_round=self.game_state.round_number,
            confidence_score=self.emotion_analyzer.get_deception_likelihood()
        )

        self._resolve_challenge(challenges, decision_data)
        return True

    def _resolve_challenge(self, challenged: bool, decision_data: DecisionData):
        """Resolve the challenge and determine card placement"""
        if not self.game_state.pending_challenge:
            return

        challenge = self.game_state.pending_challenge

        # Determine if challenger was correct
        correct = (challenged and challenge.is_bluff) or (not challenged and not challenge.is_bluff)

        # Determine who gets the card
        if correct:
            # Challenger was right, card goes to passer
            card_goes_to = challenge.passer
        else:
            # Challenger was wrong, card goes to receiver
            card_goes_to = challenge.receiver

        # Place card
        if card_goes_to == PlayerType.HUMAN:
            self.game_state.human_player.cards_on_table.append(challenge.card)
        else:
            self.game_state.ai_player.cards_on_table.append(challenge.card)

        # Update analytics
        self.analytics.add_decision(decision_data)

        # Update AI strategy learning
        if self.ai_strategy:
            outcome = {
                'correct': correct,
                'card_goes_to': card_goes_to.value,
                'challenged': challenged
            }
            decision = {
                'challenge_decision': challenged if challenge.receiver == PlayerType.AI else not challenged,
                'was_bluff': challenge.is_bluff,
                'player_type': challenge.passer.value
            }
            self.ai_strategy.update_history(self.game_state, decision, outcome)

        # Clear challenge
        self.game_state.pending_challenge = None
        self.game_state.phase = GamePhase.PLAYING

        # Create result message
        challenger_name = "AI" if challenge.receiver == PlayerType.AI else "You"
        action = "challenged" if challenged else "accepted"
        result = "correct" if correct else "wrong"
        card_recipient = "you" if card_goes_to == PlayerType.HUMAN else "AI"

        message = f"{challenger_name} {action} and was {result}! Card goes to {card_recipient}."
        self._show_message(message)

        # Check win conditions
        if self._check_win_conditions():
            return

        # Set up next turn
        self.game_state.current_player = challenge.receiver
        self.game_state.round_number += 1

        # Schedule next action
        if self.game_state.current_player == PlayerType.AI:
            self._schedule_ai_turn()
        else:
            self._show_message("Your turn! Select a card to pass to the AI.")
            self.decision_start_time = time.time()

        self._notify_state_change()

    def _check_win_conditions(self) -> bool:
        """Check if game is over and handle end game"""
        human_lost = self.game_state.human_player.has_lost()
        ai_lost = self.game_state.ai_player.has_lost()
        human_won = self.game_state.human_player.has_won()
        ai_won = self.game_state.ai_player.has_won()

        if human_lost:
            self._end_game("Game Over! You lose - you have 4 of the same creature!", False)
            return True
        elif ai_lost:
            self._end_game("Game Over! You win - AI has 4 of the same creature!", True)
            return True
        elif human_won:
            self._end_game("Game Over! You win - you played all your cards!", True)
            return True
        elif ai_won:
            self._end_game("Game Over! AI wins - it played all its cards!", False)
            return True

        return False

    def _end_game(self, message: str, human_won: bool):
        """Handle game end"""
        self.game_state.phase = GamePhase.GAME_OVER
        self.face_detector.stop_detection()

        # Update game analytics
        self.analytics.total_games += 1
        if human_won:
            self.analytics.human_wins += 1
        else:
            self.analytics.ai_wins += 1

        self._show_message(message)
        self._notify_state_change()

    def _schedule_ai_turn(self):
        """Schedule AI to take its turn"""

        def make_ai_turn():
            self._handle_ai_turn()

        self.pending_ai_action = threading.Timer(self.ai_decision_delay, make_ai_turn)
        self.pending_ai_action.start()

    def _handle_ai_turn(self):
        """AI takes its turn by passing a card"""
        if (self.game_state.phase != GamePhase.PLAYING or
                self.game_state.current_player != PlayerType.AI or
                not self.ai_strategy):
            return

        ai_hand = self.game_state.ai_player.hand
        if not ai_hand:
            return

        # AI selects card to pass
        card_to_pass = random.choice(ai_hand)

        # AI decides what to claim
        claimed_creature = self.ai_strategy.make_bluff_decision(self.game_state, card_to_pass)

        # Remove card from AI hand
        self.game_state.ai_player.hand.remove(card_to_pass)

        # Create challenge
        is_bluff = claimed_creature != card_to_pass.creature
        challenge = PendingChallenge(
            card=card_to_pass,
            claimed_creature=claimed_creature,
            is_bluff=is_bluff,
            passer=PlayerType.AI,
            receiver=PlayerType.HUMAN
        )

        self.game_state.pending_challenge = challenge
        self.game_state.phase = GamePhase.CHALLENGE

        self._show_message(f'AI passed a "{claimed_creature}" to you. Do you think it\'s telling the truth?')
        self.decision_start_time = time.time()
        self._notify_state_change()

    def update(self, dt: float):
        """Update game state (called from main loop)"""
        # This can be used for time-based updates, animations, etc.
        pass

    def cleanup(self):
        """Cleanup resources"""
        self.face_detector.stop_detection()
        if self.pending_ai_action:
            self.pending_ai_action.cancel()

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        avg_decision_time = 0.0
        if self.analytics.decision_data:
            total_time = sum(d.decision_time for d in self.analytics.decision_data)
            avg_decision_time = total_time / len(self.analytics.decision_data)

        return {
            'total_games': self.analytics.total_games,
            'human_wins': self.analytics.human_wins,
            'ai_wins': self.analytics.ai_wins,
            'win_rate': (
                        self.analytics.human_wins / self.analytics.total_games * 100) if self.analytics.total_games > 0 else 0,
            'accuracy': self.analytics.get_accuracy(),
            'bluff_detection_rate': self.analytics.get_bluff_detection_rate(),
            'avg_decision_time': avg_decision_time,
            'total_decisions': self.analytics.total_decisions,
            'ai_strategy': self.ai_strategy.name if self.ai_strategy else 'None'
        }

    def export_decision_data(self) -> List[Dict[str, Any]]:
        """Export decision data for external analysis"""
        return [decision.to_dict() for decision in self.analytics.decision_data]