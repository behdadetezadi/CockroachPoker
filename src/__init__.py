"""
Cockroach Poker - AI Bluff Detection Research Platform

A modular implementation of simplified Cockroach Poker designed for
researching AI bluff detection strategies and human behavioral analysis.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

# Import main components for easy access
from .game.game_manager import GameManager
from .game.models import GameState, Card, Player, PlayerType, GamePhase
from .ai.strategies import AIStrategy, create_strategy
from .detection.face_detection import FaceDetector, EmotionAnalyzer
from .analysis.data_analysis import BluffAnalyzer, DataExporter, MLDataPreprocessor

__all__ = [
    'GameManager',
    'GameState', 'Card', 'Player', 'PlayerType', 'GamePhase',
    'AIStrategy', 'create_strategy',
    'FaceDetector', 'EmotionAnalyzer',
    'BluffAnalyzer', 'DataExporter', 'MLDataPreprocessor'
]