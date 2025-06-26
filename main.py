#!/usr/bin/env python3
"""
Cockroach Poker - AI Bluff Detection Simulator
FIXED VERSION - Proper game rules and turn management
"""

import pygame
import sys
import random
import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import cv2
import numpy as np

# Import our AI strategies
try:
    from ai_strategies import get_strategy, get_all_strategies, AVAILABLE_STRATEGIES
except ImportError:
    print("⚠️ ai_strategies.py not found, using built-in simple AI")
    AVAILABLE_STRATEGIES = {'simple': None}

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 120, 0)
BLUE = (0, 100, 200)
RED = (200, 50, 50)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 80, 0)

# Game constants - MUCH shorter game!
CREATURES = ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']  # Only 6 creatures
CARDS_PER_CREATURE = 6  # 6 of each = 36 total cards
STARTING_HAND_SIZE = 5  # Only 5 cards each = 10 cards dealt, 26 in deck
LOSE_CONDITION = 3  # Only need 3 of same creature to lose (not 4)


class GamePhase(Enum):
    MENU = "menu"
    HUMAN_TURN = "human_turn"
    AI_TURN = "ai_turn"
    HUMAN_CHALLENGE = "human_challenge"
    AI_CHALLENGE = "ai_challenge"
    GAME_OVER = "game_over"


@dataclass
class Card:
    creature: str


@dataclass
class Player:
    name: str
    hand: List[Card]
    table_cards: List[Card]

    def has_lost(self) -> bool:
        """Check if player has 3 of same creature"""
        creature_counts = {}
        for card in self.table_cards:
            creature_counts[card.creature] = creature_counts.get(card.creature, 0) + 1
        return max(creature_counts.values(), default=0) >= LOSE_CONDITION

    def get_creature_count(self, creature: str) -> int:
        """Get count of specific creature on table"""
        return sum(1 for card in self.table_cards if card.creature == creature)


class FaceDetector:
    """Real face detection using OpenCV"""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.5
        self.detection_thread = None

        # Try to initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("✅ Camera initialized successfully")
            else:
                self.cap = None
                print("⚠️ Camera not available")
        except Exception as e:
            print(f"⚠️ Camera initialization failed: {e}")
            self.cap = None

    def start_detection(self):
        if self.cap is None:
            print("Using mock face detection")
            self._start_mock_detection()
            return

        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

    def _detection_loop(self):
        """Real face detection loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Simple emotion heuristics based on face detection
                    face = faces[0]
                    x, y, w, h = face
                    face_area = w * h

                    # Larger face = closer to camera (might be leaning in, nervous)
                    if face_area > 15000:
                        self.current_emotion = random.choice(["nervous", "focused"])
                        self.emotion_confidence = random.uniform(0.7, 0.9)
                    elif face_area < 8000:
                        self.current_emotion = random.choice(["confident", "neutral"])
                        self.emotion_confidence = random.uniform(0.6, 0.8)
                    else:
                        self.current_emotion = random.choice(["thinking", "neutral"])
                        self.emotion_confidence = random.uniform(0.5, 0.7)
                else:
                    self.current_emotion = "unknown"
                    self.emotion_confidence = 0.0

                time.sleep(0.3)  # Update every 0.3 seconds

            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(1)

    def _start_mock_detection(self):
        """Fallback mock detection"""

        def mock_loop():
            emotions = ["neutral", "confident", "nervous", "thinking", "focused"]
            while self.is_running:
                self.current_emotion = random.choice(emotions)
                self.emotion_confidence = random.uniform(0.4, 0.9)
                time.sleep(0.8)

        self.is_running = True
        self.detection_thread = threading.Thread(target=mock_loop, daemon=True)
        self.detection_thread.start()

    def get_current_state(self):
        return {
            'emotion': self.current_emotion,
            'confidence': self.emotion_confidence
        }


class SimpleAI:
    """Fallback AI if ai_strategies.py is not available"""

    def __init__(self):
        self.name = "Simple AI"
        self.description = "Basic fallback AI"

    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        return random.random() < 0.5

    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        if random.random() < 0.6:
            return actual_creature
        creatures = ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
        return random.choice([c for c in creatures if c != actual_creature])

    def learn_from_outcome(self, was_bluff: bool, opponent_called_bluff: bool, face_data: Dict, decision_time: float):
        pass


class Game:
    """Fixed game with proper rules and turn management"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Cockroach Poker - Fixed Version")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)

        # Game state
        self.phase = GamePhase.MENU
        self.human_player = Player("Human", [], [])
        self.ai_player = Player("AI", [], [])
        self.current_card = None
        self.claimed_creature = None
        self.decision_start_time = None
        self.message = "Welcome to Cockroach Poker! (Fixed Version)"
        self.message_timer = 0

        # AI and detection
        if 'simple' not in AVAILABLE_STRATEGIES:
            self.ai_strategy = get_strategy('learning')  # Default to learning AI
            self.available_strategies = list(AVAILABLE_STRATEGIES.keys())
        else:
            self.ai_strategy = SimpleAI()  # Fallback
            self.available_strategies = ['simple']

        self.current_strategy_index = 0
        self.face_detector = FaceDetector()

        # UI state
        self.selected_card_index = -1
        self.selected_claim = ""

        # Stats
        self.game_count = 0
        self.human_wins = 0
        self.ai_wins = 0

        # Start face detection
        self.face_detector.start_detection()

    def init_new_game(self):
        """Start a new game with proper setup"""
        # Create deck - smaller for quicker games
        deck = []
        for creature in CREATURES:
            for _ in range(CARDS_PER_CREATURE):
                deck.append(Card(creature))

        random.shuffle(deck)

        # Deal smaller hands for faster games
        self.human_player.hand = deck[:STARTING_HAND_SIZE]
        self.ai_player.hand = deck[STARTING_HAND_SIZE:STARTING_HAND_SIZE * 2]
        self.human_player.table_cards = []
        self.ai_player.table_cards = []

        # Human always starts
        self.phase = GamePhase.HUMAN_TURN
        self.selected_card_index = -1
        self.selected_claim = ""
        self.current_card = None
        self.claimed_creature = None

        self.set_message("Your turn! Select a card and choose what to claim it is.")

    def switch_ai_strategy(self):
        """Switch to next available AI strategy"""
        if len(self.available_strategies) <= 1:
            return

        self.current_strategy_index = (self.current_strategy_index + 1) % len(self.available_strategies)
        strategy_name = self.available_strategies[self.current_strategy_index]

        if 'simple' not in AVAILABLE_STRATEGIES:
            self.ai_strategy = get_strategy(strategy_name)
            self.set_message(f"Switched to {self.ai_strategy.name}")
        else:
            self.set_message("Only Simple AI available")

    def switch_to_strategy(self, index: int):
        """Switch to specific strategy by index"""
        if 0 <= index < len(self.available_strategies):
            self.current_strategy_index = index
            strategy_name = self.available_strategies[index]

            if 'simple' not in AVAILABLE_STRATEGIES:
                self.ai_strategy = get_strategy(strategy_name)
                self.set_message(f"Switched to {self.ai_strategy.name}")
            else:
                self.set_message("Only Simple AI available")

    def set_message(self, text: str, duration: float = 4.0):
        """Set a temporary message"""
        self.message = text
        self.message_timer = duration

    def handle_events(self):
        """Handle all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.phase == GamePhase.MENU:
                    self.init_new_game()
                elif event.key == pygame.K_r:
                    self.init_new_game()
                elif event.key == pygame.K_TAB and len(self.available_strategies) > 1:
                    self.switch_ai_strategy()
                elif event.key >= pygame.K_1 and event.key <= pygame.K_5:
                    # Quick switch to strategy 1-5
                    strategy_index = event.key - pygame.K_1
                    if strategy_index < len(self.available_strategies):
                        self.switch_to_strategy(strategy_index)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)

        return True

    def handle_mouse_click(self, pos):
        """Handle mouse clicks - ONLY during appropriate phases"""
        if self.phase == GamePhase.HUMAN_TURN:
            # Only allow interaction during human turn
            self._handle_human_turn_click(pos)
        elif self.phase == GamePhase.HUMAN_CHALLENGE:
            # Only allow challenge decision during challenge phase
            self._handle_human_challenge_click(pos)

    def _handle_human_turn_click(self, pos):
        """Handle clicks during human turn"""
        # Check card selection
        card_y = WINDOW_HEIGHT - 120
        for i, card in enumerate(self.human_player.hand):
            card_x = 50 + i * 100
            card_rect = pygame.Rect(card_x, card_y, 80, 100)
            if card_rect.collidepoint(pos):
                self.selected_card_index = i
                self.selected_claim = ""
                return

        # Check creature claim buttons
        if self.selected_card_index >= 0:
            claims_y = WINDOW_HEIGHT - 200
            for i, creature in enumerate(CREATURES):
                claim_x = 50 + i * 130
                claim_rect = pygame.Rect(claim_x, claims_y, 120, 30)
                if claim_rect.collidepoint(pos):
                    self.selected_claim = creature
                    return

            # Check pass button
            pass_rect = pygame.Rect(50, WINDOW_HEIGHT - 160, 100, 30)
            if pass_rect.collidepoint(pos) and self.selected_claim:
                self.human_pass_card()

    def _handle_human_challenge_click(self, pos):
        """Handle clicks during challenge phase"""
        challenge_rect = pygame.Rect(WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT // 2 + 70, 100, 40)
        accept_rect = pygame.Rect(WINDOW_WIDTH // 2 + 20, WINDOW_HEIGHT // 2 + 70, 100, 40)

        if challenge_rect.collidepoint(pos):
            self.human_challenge_decision(True)  # Call BLUFF
        elif accept_rect.collidepoint(pos):
            self.human_challenge_decision(False)  # Call TRUE

    def human_pass_card(self):
        """Human passes a card to AI"""
        if self.phase != GamePhase.HUMAN_TURN or self.selected_card_index < 0 or not self.selected_claim:
            return

        card = self.human_player.hand.pop(self.selected_card_index)
        self.current_card = card
        self.claimed_creature = self.selected_claim
        self.decision_start_time = time.time()

        self.set_message(f"You passed a '{self.claimed_creature}' to the AI. AI is deciding...")
        self.phase = GamePhase.AI_CHALLENGE

        # AI decides whether to challenge after a delay
        threading.Timer(random.uniform(1.5, 3.0), self.ai_challenge_decision).start()

        self.selected_card_index = -1
        self.selected_claim = ""

    def ai_challenge_decision(self):
        """AI decides whether to challenge"""
        if self.phase != GamePhase.AI_CHALLENGE:
            return

        decision_time = time.time() - self.decision_start_time if self.decision_start_time else 2.0
        face_data = self.face_detector.get_current_state()

        # Prepare game state for AI
        human_creature_counts = {}
        for card in self.human_player.table_cards:
            human_creature_counts[card.creature] = human_creature_counts.get(card.creature, 0) + 1

        game_state = {
            'human_creature_counts': human_creature_counts,
            'ai_creature_counts': {card.creature: self.ai_player.get_creature_count(card.creature) for card in
                                   self.ai_player.table_cards}
        }

        ai_challenges = self.ai_strategy.should_call_bluff(
            self.claimed_creature, game_state, face_data, decision_time
        )

        is_bluff = self.current_card.creature != self.claimed_creature

        # Resolve the challenge - CORRECT COCKROACH POKER RULES
        if ai_challenges:
            # AI says "This is a BLUFF!"
            self.set_message(f"AI calls BLUFF! The card was actually a {self.current_card.creature}")
            if is_bluff:
                # AI was CORRECT - you were lying, so YOU get the card
                self.human_player.table_cards.append(self.current_card)
                self.set_message(f"AI was RIGHT! You lied, so you get the {self.current_card.creature}")
            else:
                # AI was WRONG - you told truth, so AI gets the card
                self.ai_player.table_cards.append(self.current_card)
                self.set_message(f"AI was WRONG! You told truth, so AI gets the {self.current_card.creature}")
        else:
            # AI says "This is TRUE!"
            self.set_message(f"AI calls TRUE! The card was actually a {self.current_card.creature}")
            if is_bluff:
                # AI was WRONG - you were lying, so AI gets the card
                self.ai_player.table_cards.append(self.current_card)
                self.set_message(f"AI was WRONG! You lied, so AI gets the {self.current_card.creature}")
            else:
                # AI was CORRECT - you told truth, so YOU get the card
                self.human_player.table_cards.append(self.current_card)
                self.set_message(f"AI was RIGHT! You told truth, so you get the {self.current_card.creature}")

        # AI learns from this interaction
        self.ai_strategy.learn_from_outcome(is_bluff, ai_challenges, face_data, decision_time)

        # Check win conditions
        if self.check_game_over():
            return

        # AI's turn next
        self.phase = GamePhase.AI_TURN
        threading.Timer(2.0, self.ai_turn).start()

    def ai_turn(self):
        """AI takes its turn"""
        if self.phase != GamePhase.AI_TURN or not self.ai_player.hand:
            return

        card = random.choice(self.ai_player.hand)
        self.ai_player.hand.remove(card)

        # Prepare game state for AI decision
        human_creature_counts = {}
        for table_card in self.human_player.table_cards:
            human_creature_counts[table_card.creature] = human_creature_counts.get(table_card.creature, 0) + 1

        ai_creature_counts = {}
        for table_card in self.ai_player.table_cards:
            ai_creature_counts[table_card.creature] = ai_creature_counts.get(table_card.creature, 0) + 1

        game_state = {
            'human_creature_counts': human_creature_counts,
            'ai_creature_counts': ai_creature_counts
        }

        claimed = self.ai_strategy.choose_claim(card.creature, game_state)

        self.current_card = card
        self.claimed_creature = claimed
        self.decision_start_time = time.time()

        self.set_message(f"AI passes you a '{claimed}'. Challenge or Accept?")
        self.phase = GamePhase.HUMAN_CHALLENGE

    def human_challenge_decision(self, challenges: bool):
        """Human makes challenge decision"""
        if self.phase != GamePhase.HUMAN_CHALLENGE:
            return

        decision_time = time.time() - self.decision_start_time if self.decision_start_time else 0
        is_bluff = self.current_card.creature != self.claimed_creature
        face_data = self.face_detector.get_current_state()

        # Resolve the challenge - CORRECT COCKROACH POKER RULES
        if challenges:
            # Human says "This is a BLUFF!"
            self.set_message(f"You call BLUFF! The card was actually a {self.current_card.creature}")
            if is_bluff:
                # Human was CORRECT - AI was lying, so AI gets the card
                self.ai_player.table_cards.append(self.current_card)
                self.set_message(f"You were RIGHT! AI lied, so AI gets the {self.current_card.creature}")
            else:
                # Human was WRONG - AI told truth, so human gets the card
                self.human_player.table_cards.append(self.current_card)
                self.set_message(f"You were WRONG! AI told truth, so you get the {self.current_card.creature}")
        else:
            # Human says "This is TRUE!"
            self.set_message(f"You call TRUE! The card was actually a {self.current_card.creature}")
            if is_bluff:
                # Human was WRONG - AI was lying, so human gets the card
                self.human_player.table_cards.append(self.current_card)
                self.set_message(f"You were WRONG! AI lied, so you get the {self.current_card.creature}")
            else:
                # Human was CORRECT - AI told truth, so AI gets the card
                self.ai_player.table_cards.append(self.current_card)
                self.set_message(f"You were RIGHT! AI told truth, so AI gets the {self.current_card.creature}")

        # AI learns from this interaction
        self.ai_strategy.learn_from_outcome(is_bluff, challenges, face_data, decision_time)

        # Check win conditions
        if self.check_game_over():
            return

        # Check if anyone is out of cards
        if not self.human_player.hand and not self.ai_player.hand:
            self.phase = GamePhase.GAME_OVER
            self.set_message("All cards played! Game over!")
            return

        # Human's turn next (if they have cards)
        if self.human_player.hand:
            self.phase = GamePhase.HUMAN_TURN
            self.set_message("Your turn! Select a card and choose what to claim.")
        elif self.ai_player.hand:
            self.phase = GamePhase.AI_TURN
            threading.Timer(1.0, self.ai_turn).start()
        else:
            self.phase = GamePhase.GAME_OVER
            self.set_message("All cards played!")

    def check_game_over(self) -> bool:
        """Check if game is over"""
        if self.human_player.has_lost():
            self.phase = GamePhase.GAME_OVER
            self.game_count += 1
            self.ai_wins += 1
            self.set_message("GAME OVER! You lose - you have 3 of the same creature!")
            return True
        elif self.ai_player.has_lost():
            self.phase = GamePhase.GAME_OVER
            self.game_count += 1
            self.human_wins += 1
            self.set_message("YOU WIN! AI has 3 of the same creature!")
            return True
        return False

    def update(self, dt):
        """Update game state"""
        if self.message_timer > 0:
            self.message_timer -= dt

    def draw(self):
        """Draw everything"""
        self.screen.fill(DARK_GREEN)

        if self.phase == GamePhase.MENU:
            self.draw_menu()
        else:
            self.draw_game()

        pygame.display.flip()

    def draw_menu(self):
        """Draw main menu"""
        title = self.big_font.render("Cockroach Poker - FIXED VERSION", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)

        subtitle = self.font.render("Real face detection • Smart AI • Proper game rules", True, YELLOW)
        sub_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60))
        self.screen.blit(subtitle, sub_rect)

        instruction = self.font.render("Press SPACE to start", True, WHITE)
        inst_rect = instruction.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(instruction, inst_rect)

        # Rules explanation
        rules = [
            "RULES:",
            "• Claim what creature you're passing (truth or lie!)",
            "• Opponent calls 'TRUE!' or 'BLUFF!'",
            "• Whoever is WRONG gets the card",
            "• Get 3 of same creature = YOU LOSE!"
        ]

        y_start = WINDOW_HEIGHT // 2 + 50
        for i, rule in enumerate(rules):
            color = YELLOW if i == 0 else LIGHT_GRAY
            rule_surface = self.font.render(rule, True, color)
            rule_rect = rule_surface.get_rect(center=(WINDOW_WIDTH // 2, y_start + i * 25))
            self.screen.blit(rule_surface, rule_rect)

        # Show stats if any games played
        if self.game_count > 0:
            stats = self.font.render(
                f"Games: {self.game_count} | Your Wins: {self.human_wins} | AI Wins: {self.ai_wins}", True, LIGHT_GRAY)
            stats_rect = stats.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 180))
            self.screen.blit(stats, stats_rect)

        # Show current AI strategy and controls
        if len(self.available_strategies) > 1:
            strategy_text = f"Current AI: {self.ai_strategy.name}"
            strategy_surface = self.font.render(strategy_text, True, YELLOW)
            strategy_rect = strategy_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
            self.screen.blit(strategy_surface, strategy_rect)

            controls_text = "Press TAB to switch AI | Press 1-5 for specific strategy"
            controls_surface = self.font.render(controls_text, True, LIGHT_GRAY)
            controls_rect = controls_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 70))
            self.screen.blit(controls_surface, controls_rect)

            # Show available strategies
            strategies_text = " | ".join(
                [f"{i + 1}:{name.title()}" for i, name in enumerate(self.available_strategies)])
            strategies_surface = self.font.render(strategies_text, True, LIGHT_GRAY)
            strategies_rect = strategies_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 40))
            self.screen.blit(strategies_surface, strategies_rect)

    def draw_game(self):
        """Draw game screen"""
        # Phase indicator
        phase_color = WHITE
        if self.phase == GamePhase.HUMAN_TURN:
            phase_text = "YOUR TURN"
            phase_color = YELLOW
        elif self.phase == GamePhase.AI_TURN:
            phase_text = "AI TURN"
            phase_color = LIGHT_GRAY
        elif self.phase == GamePhase.HUMAN_CHALLENGE:
            phase_text = "YOUR DECISION"
            phase_color = RED
        elif self.phase == GamePhase.AI_CHALLENGE:
            phase_text = "AI DECIDING"
            phase_color = BLUE
        else:
            phase_text = "GAME OVER"
            phase_color = WHITE

        phase_surface = self.big_font.render(phase_text, True, phase_color)
        phase_rect = phase_surface.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(phase_surface, phase_rect)

        # AI area
        ai_text = self.font.render(f"AI Cards on Table ({len(self.ai_player.table_cards)}):", True, WHITE)
        self.screen.blit(ai_text, (50, 80))

        for i, card in enumerate(self.ai_player.table_cards):
            card_rect = pygame.Rect(50 + i * 60, 110, 50, 70)
            pygame.draw.rect(self.screen, WHITE, card_rect)
            pygame.draw.rect(self.screen, BLACK, card_rect, 2)

            creature_text = self.font.render(card.creature[:3], True, BLACK)
            text_rect = creature_text.get_rect(center=card_rect.center)
            self.screen.blit(creature_text, text_rect)

        # Show AI creature counts
        ai_counts = {}
        for card in self.ai_player.table_cards:
            ai_counts[card.creature] = ai_counts.get(card.creature, 0) + 1

        y_offset = 200
        for creature, count in ai_counts.items():
            color = RED if count >= LOSE_CONDITION else YELLOW if count >= 2 else WHITE
            count_text = self.font.render(f"{creature}: {count}", True, color)
            self.screen.blit(count_text, (500, y_offset))
            y_offset += 25

        # AI hand
        ai_hand_text = self.font.render(f"AI Hand: {len(self.ai_player.hand)} cards", True, WHITE)
        self.screen.blit(ai_hand_text, (50, 200))

        # Human table cards
        human_text = self.font.render(f"Your Cards on Table ({len(self.human_player.table_cards)}):", True, WHITE)
        self.screen.blit(human_text, (50, 350))

        for i, card in enumerate(self.human_player.table_cards):
            card_rect = pygame.Rect(50 + i * 60, 380, 50, 70)
            pygame.draw.rect(self.screen, WHITE, card_rect)
            pygame.draw.rect(self.screen, BLACK, card_rect, 2)

            creature_text = self.font.render(card.creature[:3], True, BLACK)
            text_rect = creature_text.get_rect(center=card_rect.center)
            self.screen.blit(creature_text, text_rect)

        # Show human creature counts
        human_counts = {}
        for card in self.human_player.table_cards:
            human_counts[card.creature] = human_counts.get(card.creature, 0) + 1

        y_offset = 350
        for creature, count in human_counts.items():
            color = RED if count >= LOSE_CONDITION else YELLOW if count >= 2 else WHITE
            count_text = self.font.render(f"{creature}: {count}", True, color)
            self.screen.blit(count_text, (500, y_offset))
            y_offset += 25

        # Human hand (only interactive during human turn)
        if self.phase == GamePhase.HUMAN_TURN:
            hand_text = self.font.render("Your Hand - Click to select:", True, YELLOW)
            self.screen.blit(hand_text, (50, WINDOW_HEIGHT - 150))

            for i, card in enumerate(self.human_player.hand):
                card_x = 50 + i * 100
                card_y = WINDOW_HEIGHT - 120
                card_rect = pygame.Rect(card_x, card_y, 80, 100)

                color = YELLOW if i == self.selected_card_index else WHITE
                pygame.draw.rect(self.screen, color, card_rect)
                pygame.draw.rect(self.screen, BLACK, card_rect, 2)

                creature_text = self.font.render(card.creature, True, BLACK)
                text_rect = creature_text.get_rect(center=card_rect.center)
                self.screen.blit(creature_text, text_rect)

            # Claim buttons
            if self.selected_card_index >= 0:
                claim_text = self.font.render("Claim this card is a:", True, WHITE)
                self.screen.blit(claim_text, (50, WINDOW_HEIGHT - 230))

                for i, creature in enumerate(CREATURES):
                    claim_x = 50 + i * 130
                    claim_y = WINDOW_HEIGHT - 200
                    claim_rect = pygame.Rect(claim_x, claim_y, 120, 30)

                    color = YELLOW if creature == self.selected_claim else LIGHT_GRAY
                    pygame.draw.rect(self.screen, color, claim_rect)
                    pygame.draw.rect(self.screen, BLACK, claim_rect, 2)

                    text = self.font.render(creature, True, BLACK)
                    text_rect = text.get_rect(center=claim_rect.center)
                    self.screen.blit(text, text_rect)

                # Pass button
                if self.selected_claim:
                    pass_rect = pygame.Rect(50, WINDOW_HEIGHT - 160, 100, 30)
                    pygame.draw.rect(self.screen, BLUE, pass_rect)
                    pygame.draw.rect(self.screen, BLACK, pass_rect, 2)

                    pass_text = self.font.render("PASS CARD", True, WHITE)
                    text_rect = pass_text.get_rect(center=pass_rect.center)
                    self.screen.blit(pass_text, text_rect)

        elif self.phase == GamePhase.HUMAN_CHALLENGE:
            # Challenge decision
            challenge_text = self.big_font.render(f"AI claims: {self.claimed_creature}", True, WHITE)
            text_rect = challenge_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(challenge_text, text_rect)

            instruction = self.font.render("Is the AI telling the truth or bluffing?", True, WHITE)
            inst_rect = instruction.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30))
            self.screen.blit(instruction, inst_rect)

            # Buttons
            challenge_rect = pygame.Rect(WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT // 2 + 70, 100, 40)
            accept_rect = pygame.Rect(WINDOW_WIDTH // 2 + 20, WINDOW_HEIGHT // 2 + 70, 100, 40)

            pygame.draw.rect(self.screen, RED, challenge_rect)
            pygame.draw.rect(self.screen, BLUE, accept_rect)
            pygame.draw.rect(self.screen, BLACK, challenge_rect, 2)
            pygame.draw.rect(self.screen, BLACK, accept_rect, 2)

            challenge_text = self.font.render("BLUFF!", True, WHITE)
            accept_text = self.font.render("TRUE!", True, WHITE)

            chal_rect = challenge_text.get_rect(center=challenge_rect.center)
            acc_rect = accept_text.get_rect(center=accept_rect.center)

            self.screen.blit(challenge_text, chal_rect)
            self.screen.blit(accept_text, acc_rect)

        # Face detection info
        face_data = self.face_detector.get_current_state()
        face_text = self.font.render(f"Emotion: {face_data['emotion']} ({face_data['confidence']:.2f})", True, WHITE)
        self.screen.blit(face_text, (WINDOW_WIDTH - 350, 80))

        # AI strategy info
        ai_text = self.font.render(f"AI: {self.ai_strategy.name}", True, YELLOW)
        self.screen.blit(ai_text, (WINDOW_WIDTH - 350, 110))

        # AI description
        if hasattr(self.ai_strategy, 'description'):
            desc_text = self.font.render(self.ai_strategy.description[:30] + "...", True, LIGHT_GRAY)
            self.screen.blit(desc_text, (WINDOW_WIDTH - 350, 130))

        # AI learning stats (if available)
        if hasattr(self.ai_strategy, 'player_stats') and self.ai_strategy.player_stats['total_claims'] > 0:
            bluff_rate = self.ai_strategy.player_stats['total_bluffs'] / self.ai_strategy.player_stats['total_claims']
            stats_text = self.font.render(f"Your Bluff Rate: {bluff_rate:.1%}", True, WHITE)
            self.screen.blit(stats_text, (WINDOW_WIDTH - 350, 160))

            avg_time = self.ai_strategy.player_stats['avg_decision_time']
            time_text = self.font.render(f"Avg Decision: {avg_time:.1f}s", True, WHITE)
            self.screen.blit(time_text, (WINDOW_WIDTH - 350, 180))

            # Show AI's recent accuracy if available
            if hasattr(self.ai_strategy, 'player_stats') and 'recent_accuracy' in self.ai_strategy.player_stats:
                accuracy = self.ai_strategy.player_stats['recent_accuracy']
                acc_text = self.font.render(f"AI Accuracy: {accuracy:.1%}", True, WHITE)
                self.screen.blit(acc_text, (WINDOW_WIDTH - 350, 200))

        # Strategy switching controls
        if len(self.available_strategies) > 1:
            switch_text = self.font.render("TAB: Switch AI", True, GRAY)
            self.screen.blit(switch_text, (WINDOW_WIDTH - 350, 230))

        # Message
        if self.message and self.message_timer > 0:
            msg_surface = self.font.render(self.message, True, WHITE)
            msg_rect = msg_surface.get_rect(center=(WINDOW_WIDTH // 2, 60))
            pygame.draw.rect(self.screen, BLACK, msg_rect.inflate(20, 10))
            self.screen.blit(msg_surface, msg_rect)

        # Instructions
        if self.phase == GamePhase.GAME_OVER:
            restart_text = self.font.render("Press R to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
            self.screen.blit(restart_text, restart_rect)

    def run(self):
        """Main game loop"""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            running = self.handle_events()
            self.update(dt)
            self.draw()

        self.face_detector.stop_detection()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        pygame.quit()
        sys.exit()