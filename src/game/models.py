"""
Game data models and structures
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import time


class GamePhase(Enum):
    SETUP = "setup"
    PLAYING = "playing"
    CHALLENGE = "challenge"
    GAME_OVER = "game_over"


class PlayerType(Enum):
    HUMAN = "human"
    AI = "ai"


@dataclass
class Card:
    """Represents a single game card"""
    id: str
    creature: str

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Player:
    """Represents a player in the game"""
    player_type: PlayerType
    hand: List[Card] = field(default_factory=list)
    cards_on_table: List[Card] = field(default_factory=list)
    score: int = 0

    def has_lost(self) -> bool:
        """Check if player has lost (4 of same creature)"""
        creature_counts = {}
        for card in self.cards_on_table:
            creature_counts[card.creature] = creature_counts.get(card.creature, 0) + 1
        return max(creature_counts.values()) >= 4 if creature_counts else False

    def has_won(self) -> bool:
        """Check if player has won (empty hand)"""
        return len(self.hand) == 0


@dataclass
class PendingChallenge:
    """Represents a pending challenge situation"""
    card: Card
    claimed_creature: str
    is_bluff: bool
    passer: PlayerType
    receiver: PlayerType
    timestamp: float = field(default_factory=time.time)


@dataclass
class GameState:
    """Main game state container"""
    human_player: Player = field(default_factory=lambda: Player(PlayerType.HUMAN))
    ai_player: Player = field(default_factory=lambda: Player(PlayerType.AI))
    current_player: PlayerType = PlayerType.HUMAN
    phase: GamePhase = GamePhase.SETUP
    round_number: int = 1
    pending_challenge: Optional[PendingChallenge] = None
    last_action_message: str = ""

    def get_current_player(self) -> Player:
        """Get the current player object"""
        return self.human_player if self.current_player == PlayerType.HUMAN else self.ai_player

    def get_opponent(self, player_type: PlayerType) -> Player:
        """Get the opponent of the specified player"""
        return self.ai_player if player_type == PlayerType.HUMAN else self.human_player

    def switch_current_player(self):
        """Switch to the other player"""
        self.current_player = PlayerType.AI if self.current_player == PlayerType.HUMAN else PlayerType.HUMAN


@dataclass
class DecisionData:
    """Data collected during a decision"""
    timestamp: float
    claimed_creature: str
    actual_creature: str
    was_bluff: bool
    player_challenged: bool
    decision_time: float
    facial_expression: Optional[Dict[str, Any]] = None
    game_round: int = 0
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis"""
        return {
            'timestamp': self.timestamp,
            'claimed_creature': self.claimed_creature,
            'actual_creature': self.actual_creature,
            'was_bluff': self.was_bluff,
            'player_challenged': self.player_challenged,
            'decision_time': self.decision_time,
            'facial_expression': self.facial_expression,
            'game_round': self.game_round,
            'confidence_score': self.confidence_score
        }


@dataclass
class FacialExpression:
    """Facial expression detection data"""
    emotion: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class GameAnalytics:
    """Game analytics and metrics"""
    total_games: int = 0
    human_wins: int = 0
    ai_wins: int = 0
    total_decisions: int = 0
    correct_challenges: int = 0
    bluffs_detected: int = 0
    total_bluffs: int = 0
    avg_decision_time: float = 0.0
    decision_data: List[DecisionData] = field(default_factory=list)

    def add_decision(self, decision: DecisionData):
        """Add a new decision to analytics"""
        self.decision_data.append(decision)
        self.total_decisions += 1

        if (decision.player_challenged and decision.was_bluff) or \
                (not decision.player_challenged and not decision.was_bluff):
            self.correct_challenges += 1

        if decision.was_bluff:
            self.total_bluffs += 1
            if decision.player_challenged:
                self.bluffs_detected += 1

    def get_accuracy(self) -> float:
        """Get challenge accuracy percentage"""
        return (self.correct_challenges / self.total_decisions * 100) if self.total_decisions > 0 else 0.0

    def get_bluff_detection_rate(self) -> float:
        """Get bluff detection rate percentage"""
        return (self.bluffs_detected / self.total_bluffs * 100) if self.total_bluffs > 0 else 0.0