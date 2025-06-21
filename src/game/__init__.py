"""
Game logic and state management module

Contains the core game mechanics, data models, and game state management
for the Cockroach Poker simulation.
"""

from .game_manager import GameManager
from .models import (
    Card, Player, GameState, PendingChallenge, DecisionData,
    GamePhase, PlayerType, GameAnalytics, FacialExpression
)

__all__ = [
    'GameManager',
    'Card', 'Player', 'GameState', 'PendingChallenge', 'DecisionData',
    'GamePhase', 'PlayerType', 'GameAnalytics', 'FacialExpression'
]