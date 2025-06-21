"""
Game configuration constants
"""

# Window settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60
WINDOW_TITLE = "Cockroach Poker - AI Bluff Detection Lab"

# Game constants
CREATURES = ['cockroach', 'rat', 'stinkbug', 'fly', 'spider', 'scorpion', 'toad', 'bat']
CARDS_PER_CREATURE = 8
MAX_SAME_CREATURE = 4
HAND_SIZE = 8

# UI constants
CARD_WIDTH = 80
CARD_HEIGHT = 120
CARD_MARGIN = 10
BUTTON_HEIGHT = 40
FONT_SIZE_LARGE = 24
FONT_SIZE_MEDIUM = 18
FONT_SIZE_SMALL = 14

# Colors (RGB)
COLORS = {
    'background': (20, 40, 20),
    'card_bg': (60, 60, 60),
    'card_selected': (100, 150, 255),
    'card_hover': (80, 80, 80),
    'button_normal': (70, 130, 180),
    'button_hover': (100, 160, 210),
    'button_disabled': (100, 100, 100),
    'text_white': (255, 255, 255),
    'text_gray': (200, 200, 200),
    'text_dark': (50, 50, 50),
    'panel_bg': (40, 40, 40),
    'panel_border': (80, 80, 80),
    'success': (50, 200, 50),
    'warning': (255, 200, 50),
    'danger': (200, 50, 50),
    'info': (50, 150, 255)
}

# Creature emojis mapping
CREATURE_EMOJIS = {
    'cockroach': '🪳',
    'rat': '🐀',
    'stinkbug': '🐛',
    'fly': '🪰',
    'spider': '🕷️',
    'scorpion': '🦂',
    'toad': '🐸',
    'bat': '🦇'
}

# AI Strategy types
AI_STRATEGIES = {
    'random': 'Random AI',
    'pattern': 'Pattern-Based AI',
    'reinforcement': 'RL Agent',
    'neural': 'Neural Network AI'
}

# Data collection settings
COLLECT_FACE_DATA = True
COLLECT_TIMING_DATA = True
COLLECT_PATTERN_DATA = True

# Face detection settings (mock)
FACE_DETECTION_UPDATE_RATE = 1.0  # seconds
EXPRESSION_CONFIDENCE_THRESHOLD = 0.7