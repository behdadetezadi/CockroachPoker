import pygame
import sys
import json
import random
import os
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Initialize Pygame
pygame.init()

# Game Configuration
GAME_CONFIG = {
    'hand_size': 5,
    'animals': ['fly', 'rat', 'toad', 'bat'],
    'lose_threshold': 3,
    'cards_per_animal': 8
}

# UI Configuration
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
RED = (220, 20, 60)
BLUE = (70, 130, 180)
GOLD = (255, 215, 0)
PURPLE = (147, 112, 219)
BACKGROUND = (45, 52, 64)  # Dark blue-gray
CARD_BG = (236, 239, 244)  # Light gray for cards
MESSAGE_BG = (59, 66, 82)  # Darker gray for message boxes
TEXT_LIGHT = (236, 239, 244)  # Light text

# Fonts
TITLE_FONT = pygame.font.Font(None, 48)
LARGE_FONT = pygame.font.Font(None, 36)
MEDIUM_FONT = pygame.font.Font(None, 28)
SMALL_FONT = pygame.font.Font(None, 22)

# Card dimensions
CARD_WIDTH = 100
CARD_HEIGHT = 140
CARD_SPACING = 15


class GameState:
    """Represents the complete state of the game"""

    def __init__(self, config: Dict):
        self.config = config
        self.deck = self._create_deck()
        self.player_hand = []
        self.ai_hand = []
        self.player_face_up = []
        self.ai_face_up = []
        self.claiming_player = 'player'
        self.current_claim = None
        self.current_card = None
        self.phase = 'claim'
        self.game_over = False
        self.winner = None

    def _create_deck(self) -> List[str]:
        deck = []
        for animal in self.config['animals']:
            deck.extend([animal] * self.config['cards_per_animal'])
        random.shuffle(deck)
        return deck

    def deal_hands(self):
        hand_size = self.config['hand_size']
        self.player_hand = [self.deck.pop() for _ in range(hand_size)]
        self.ai_hand = [self.deck.pop() for _ in range(hand_size)]

    def check_loss_condition(self) -> Optional[str]:
        threshold = self.config['lose_threshold']

        player_counts = defaultdict(int)
        for card in self.player_face_up:
            player_counts[card] += 1
            if player_counts[card] >= threshold:
                return 'ai'

        ai_counts = defaultdict(int)
        for card in self.ai_face_up:
            ai_counts[card] += 1
            if ai_counts[card] >= threshold:
                return 'player'

        return None

    def check_empty_hands(self) -> Optional[str]:
        if not self.player_hand and not self.ai_hand:
            player_count = len(self.player_face_up)
            ai_count = len(self.ai_face_up)

            if player_count > ai_count:
                return 'ai'
            elif ai_count > player_count:
                return 'player'
            else:
                return 'tie'
        return None

    def switch_claiming_player(self):
        self.claiming_player = 'ai' if self.claiming_player == 'player' else 'player'

    def to_dict(self) -> Dict:
        return {
            'player_hand': self.player_hand.copy(),
            'ai_hand': self.ai_hand.copy(),
            'player_face_up': self.player_face_up.copy(),
            'ai_face_up': self.ai_face_up.copy(),
            'claiming_player': self.claiming_player,
            'current_claim': self.current_claim,
            'current_card': self.current_card,
            'phase': self.phase,
            'game_over': self.game_over,
            'winner': self.winner
        }


class DataLogger:
    """Logs game data for training purposes"""

    def __init__(self, log_dir: str = 'game_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_game_log = []
        self.game_count = len([f for f in os.listdir(log_dir) if f.endswith('.json')])

    def log_state_action(self, state: Dict, action: Dict, reward: float = 0):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': action,
            'reward': reward,
            'turn_number': len(self.current_game_log)
        }
        self.current_game_log.append(entry)

    def save_game(self):
        if self.current_game_log:
            filename = f"game_{self.game_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(self.current_game_log, f, indent=2)
            self.current_game_log = []
            self.game_count += 1

    def load_all_games(self) -> List[List[Dict]]:
        games = []
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.log_dir, filename)
                with open(filepath, 'r') as f:
                    games.append(json.load(f))
        return games


class Button:
    """Interactive UI button"""

    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)

        text_surf = MEDIUM_FONT.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.current_color = self.hover_color if self.rect.collidepoint(event.pos) else self.color
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Card:
    """Visual card representation"""

    def __init__(self, animal, x, y):
        self.animal = animal
        self.rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        self.hovered = False

    def draw(self, screen, face_up=True, selected=False):
        # Card background
        if selected:
            color = GOLD
        elif self.hovered:
            color = (220, 220, 220)
        else:
            color = CARD_BG

        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 3, border_radius=8)

        if face_up:
            # Draw animal name in center
            name = self.animal.upper()
            if len(name) > 7:
                # Split long names into two lines
                mid = len(name) // 2
                line1 = name[:mid]
                line2 = name[mid:]

                text1 = SMALL_FONT.render(line1, True, BLACK)
                text2 = SMALL_FONT.render(line2, True, BLACK)

                rect1 = text1.get_rect(center=(self.rect.centerx, self.rect.centery - 15))
                rect2 = text2.get_rect(center=(self.rect.centerx, self.rect.centery + 15))

                screen.blit(text1, rect1)
                screen.blit(text2, rect2)
            else:
                text = MEDIUM_FONT.render(name, True, BLACK)
                text_rect = text.get_rect(center=self.rect.center)
                screen.blit(text, text_rect)
        else:
            # Card back
            inner_rect = self.rect.inflate(-10, -10)
            pygame.draw.rect(screen, PURPLE, inner_rect, border_radius=6)

            # Pattern on card back
            for i in range(3):
                for j in range(4):
                    x = inner_rect.x + 15 + i * 25
                    y = inner_rect.y + 20 + j * 30
                    pygame.draw.circle(screen, (180, 150, 255), (x, y), 8)

            back_text = LARGE_FONT.render('?', True, WHITE)
            back_rect = back_text.get_rect(center=self.rect.center)
            screen.blit(back_text, back_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class CockroachPokerUI:
    """Main game UI and logic controller"""

    def __init__(self, ai_agent=None):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Cockroach Poker - RL Edition")
        self.clock = pygame.time.Clock()

        self.state = GameState(GAME_CONFIG)
        self.logger = DataLogger()
        self.agent = ai_agent  # AI agent injected from outside

        self.state.deal_hands()
        self.state.claiming_player = random.choice(['player', 'ai'])

        self.selected_card = None
        self.selected_claim = None
        self.message = ""
        self.message_color = BLACK
        self.sub_message = ""

        self.player_cards = []
        self.ai_cards = []
        self.claim_buttons = []
        self.response_buttons = []

        self.setup_ui()

    def setup_ui(self):
        """Initialize UI buttons"""
        # Create claim buttons
        button_width = 120
        button_height = 40
        start_x = (SCREEN_WIDTH - (len(GAME_CONFIG['animals'][:4]) * (button_width + 10))) // 2

        for i, animal in enumerate(GAME_CONFIG['animals'][:4]):
            x = start_x + i * (button_width + 10)
            y = SCREEN_HEIGHT // 2 - 40
            btn = Button(x, y, button_width, button_height, animal.capitalize(), BLUE, DARK_GREEN)
            self.claim_buttons.append((animal, btn))

        # Second row of animals
        start_x = (SCREEN_WIDTH - (len(GAME_CONFIG['animals'][4:]) * (button_width + 10))) // 2
        for i, animal in enumerate(GAME_CONFIG['animals'][4:]):
            x = start_x + i * (button_width + 10)
            y = SCREEN_HEIGHT // 2 + 10
            btn = Button(x, y, button_width, button_height, animal.capitalize(), BLUE, DARK_GREEN)
            self.claim_buttons.append((animal, btn))

        # Response buttons
        truth_btn = Button(SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 - 20, 120, 60, "TRUTH", GREEN, DARK_GREEN)
        bluff_btn = Button(SCREEN_WIDTH // 2 + 10, SCREEN_HEIGHT // 2 - 20, 120, 60, "BLUFF", RED, DARK_GRAY)
        self.response_buttons = [('truth', truth_btn), ('bluff', bluff_btn)]

    def update_card_positions(self):
        """Update card positions based on current hands"""
        # Player hand - positioned at bottom with more spacing from edge
        self.player_cards = []
        total_width = len(self.state.player_hand) * (CARD_WIDTH + CARD_SPACING) - CARD_SPACING
        start_x = (SCREEN_WIDTH - total_width) // 2
        y = SCREEN_HEIGHT - CARD_HEIGHT - 30

        for i, animal in enumerate(self.state.player_hand):
            x = start_x + i * (CARD_WIDTH + CARD_SPACING)
            card = Card(animal, x, y)
            self.player_cards.append(card)

        # AI cards (face down) - positioned at top with more spacing
        self.ai_cards = []
        total_width = len(self.state.ai_hand) * (CARD_WIDTH + CARD_SPACING) - CARD_SPACING
        start_x = (SCREEN_WIDTH - total_width) // 2
        y = 90

        for i in range(len(self.state.ai_hand)):
            x = start_x + i * (CARD_WIDTH + CARD_SPACING)
            card = Card('unknown', x, y)
            self.ai_cards.append(card)

    def draw_face_up_cards(self):
        """Draw face-up penalty cards for both players"""
        # Player face-up cards
        if self.state.player_face_up:
            y = SCREEN_HEIGHT - 230

            # Background panel
            panel_rect = pygame.Rect(10, y - 40, 600, 80)
            pygame.draw.rect(self.screen, MESSAGE_BG, panel_rect, border_radius=10)
            pygame.draw.rect(self.screen, TEXT_LIGHT, panel_rect, 2, border_radius=10)

            text = MEDIUM_FONT.render("Your Penalty Cards:", True, TEXT_LIGHT)
            self.screen.blit(text, (20, y - 35))

            card_counts = defaultdict(int)
            for card in self.state.player_face_up:
                card_counts[card] += 1

            x = 20
            for animal, count in sorted(card_counts.items()):
                card_text = SMALL_FONT.render(f"{animal}: {count}", True, TEXT_LIGHT)

                # Highlight if at threshold
                if count >= GAME_CONFIG['lose_threshold']:
                    pygame.draw.rect(self.screen, RED, (x - 5, y - 5, 110, 30), 3, border_radius=5)

                self.screen.blit(card_text, (x, y))
                x += 120

        # AI face-up cards
        if self.state.ai_face_up:
            y = 200

            # Background panel
            panel_rect = pygame.Rect(10, y - 40, 600, 80)
            pygame.draw.rect(self.screen, MESSAGE_BG, panel_rect, border_radius=10)
            pygame.draw.rect(self.screen, TEXT_LIGHT, panel_rect, 2, border_radius=10)

            text = MEDIUM_FONT.render("AI Penalty Cards:", True, TEXT_LIGHT)
            self.screen.blit(text, (20, y - 35))

            card_counts = defaultdict(int)
            for card in self.state.ai_face_up:
                card_counts[card] += 1

            x = 20
            for animal, count in sorted(card_counts.items()):
                card_text = SMALL_FONT.render(f"{animal}: {count}", True, TEXT_LIGHT)

                if count >= GAME_CONFIG['lose_threshold']:
                    pygame.draw.rect(self.screen, RED, (x - 5, y - 5, 110, 30), 3, border_radius=5)

                self.screen.blit(card_text, (x, y))
                x += 120

    def draw(self):
        """Render the entire game screen"""
        self.screen.fill(BACKGROUND)

        # Title with background
        title_bg = pygame.Rect(0, 0, SCREEN_WIDTH, 80)
        pygame.draw.rect(self.screen, MESSAGE_BG, title_bg)
        pygame.draw.rect(self.screen, PURPLE, (0, 78, SCREEN_WIDTH, 2))

        title = TITLE_FONT.render("Cockroach Poker", True, PURPLE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)

        # Message area - positioned in center, not overlapping with cards
        if self.message:
            msg_y = SCREEN_HEIGHT // 2 - 100

            msg_surf = LARGE_FONT.render(self.message, True, TEXT_LIGHT)
            msg_rect = msg_surf.get_rect(center=(SCREEN_WIDTH // 2, msg_y))

            # Background for message
            bg_rect = msg_rect.inflate(60, 30)
            pygame.draw.rect(self.screen, MESSAGE_BG, bg_rect, border_radius=12)
            pygame.draw.rect(self.screen, self.message_color, bg_rect, 3, border_radius=12)

            self.screen.blit(msg_surf, msg_rect)

            if self.sub_message:
                sub_surf = MEDIUM_FONT.render(self.sub_message, True, TEXT_LIGHT)
                sub_rect = sub_surf.get_rect(center=(SCREEN_WIDTH // 2, msg_y + 40))
                self.screen.blit(sub_surf, sub_rect)

        # Draw cards
        self.update_card_positions()

        for card in self.ai_cards:
            card.draw(self.screen, face_up=False)

        for i, card in enumerate(self.player_cards):
            selected = (self.selected_card == i)
            card.draw(self.screen, face_up=True, selected=selected)

        # Draw face-up cards
        self.draw_face_up_cards()

        # Draw claim buttons if in claim selection phase
        if self.state.phase == 'claim' and self.state.claiming_player == 'player' and self.selected_card is not None and self.selected_claim is None:
            label = LARGE_FONT.render("Choose what to claim:", True, TEXT_LIGHT)
            label_rect = label.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 140))

            # Label background
            label_bg = label_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, MESSAGE_BG, label_bg, border_radius=10)
            pygame.draw.rect(self.screen, GOLD, label_bg, 2, border_radius=10)

            self.screen.blit(label, label_rect)

            for animal, btn in self.claim_buttons:
                btn.draw(self.screen)

        # Draw response buttons if in response phase
        if self.state.phase == 'respond' and self.state.claiming_player == 'ai':
            for response, btn in self.response_buttons:
                btn.draw(self.screen)

        pygame.display.flip()

    def handle_player_claim(self, event):
        """Handle player's claiming turn"""
        if self.selected_card is None:
            # Select card
            for i, card in enumerate(self.player_cards):
                if card.handle_event(event):
                    self.selected_card = i
                    self.message = "Card selected! Now choose what to claim:"
                    self.message_color = BLUE
                    return
        else:
            # Select claim
            for animal, btn in self.claim_buttons:
                if btn.handle_event(event):
                    card = self.state.player_hand.pop(self.selected_card)
                    self.state.current_card = card
                    self.state.current_claim = animal
                    self.state.phase = 'respond'

                    self.message = f"You claim the card is a {animal}"
                    self.sub_message = "AI is thinking..."
                    self.message_color = BLACK

                    self.logger.log_state_action(
                        self.state.to_dict(),
                        {'player': 'player', 'card': card, 'claim': animal, 'action': 'claim'}
                    )

                    self.selected_card = None
                    self.selected_claim = None

                    # Schedule AI response
                    pygame.time.set_timer(pygame.USEREVENT + 1, 1500)
                    return

    def ai_respond(self):
        """Handle AI's response to player's claim - NO CHEATING!"""

        # Create observable state WITHOUT the actual card
        observable_state = {
            'ai_hand': self.state.ai_hand,
            'player_face_up': self.state.player_face_up,
            'ai_face_up': self.state.ai_face_up,
            'current_claim': self.state.current_claim,
            # NO current_card - AI can't see it yet!
        }

        # AI makes decision based only on observable info
        if self.agent:
            response = self.agent.choose_response(observable_state)
        else:
            response = random.choice(['truth', 'bluff'])

        # NOW reveal the card and calculate outcome
        card = self.state.current_card
        claim = self.state.current_claim
        is_truth = (card == claim)

        # FIXED REWARD SYSTEM:
        # Correct guess = GOOD (+1)
        # Wrong guess = BAD (-1)
        if (response == 'truth' and is_truth) or (response == 'bluff' and not is_truth):
            # AI guessed CORRECTLY
            self.state.player_face_up.append(card)
            self.message = f"AI calls {response.upper()}! Correct! The card was a {card}"
            self.sub_message = "Card goes to you"
            reward = +1  # ✅ FIXED: Reward correct guessing
        else:
            # AI guessed WRONG
            self.state.ai_face_up.append(card)
            self.message = f"AI calls {response.upper()}! Wrong! The card was a {card}"
            self.sub_message = "Card goes to AI"
            reward = -1  # ✅ FIXED: Penalize wrong guessing

        # Check for terminal state (game over)
        winner = self.state.check_loss_condition()
        is_terminal = False

        if winner == 'ai':
            reward += 50  # ✅ BIG bonus for winning
            is_terminal = True
        elif winner == 'player':
            reward -= 50  # ✅ BIG penalty for losing
            is_terminal = True

        # Update agent's belief state (POMDP)
        if self.agent and hasattr(self.agent, 'update_belief'):
            # Estimate cards remaining (simplified - could be more accurate)
            cards_remaining = defaultdict(int)
            for animal in GAME_CONFIG['animals']:
                cards_remaining[animal] = 1  # Simplified
            self.agent.update_belief(claim, card, cards_remaining)

        # Log with observable state + outcome for training
        log_state = observable_state.copy()
        log_state['current_card'] = card  # Include for training analysis
        log_state['game_over'] = is_terminal

        self.logger.log_state_action(
            log_state,
            {'player': 'ai', 'response': response, 'action': 'respond'},
            reward
        )

        self.state.current_card = None
        self.state.current_claim = None
        self.state.phase = 'claim'
        self.state.switch_claiming_player()

        # Check for game over
        if not self.check_game_over():
            # Schedule AI claim if it's AI's turn
            if self.state.claiming_player == 'ai' and self.state.ai_hand:
                pygame.time.set_timer(pygame.USEREVENT + 2, 2000)

    def ai_claim(self):
        """Handle AI's claiming turn"""
        if not self.state.ai_hand:
            self.state.switch_claiming_player()
            return

        if self.agent:
            card, claim = self.agent.choose_card_and_claim(self.state.ai_hand)
        else:
            card = random.choice(self.state.ai_hand)
            claim = random.choice(GAME_CONFIG['animals'])

        if card:
            self.state.ai_hand.remove(card)
            self.state.current_card = card
            self.state.current_claim = claim
            self.state.phase = 'respond'

            self.message = f"AI claims the card is a {claim}"
            self.sub_message = "Truth or Bluff?"
            self.message_color = BLACK

            self.logger.log_state_action(
                self.state.to_dict(),
                {'player': 'ai', 'claim': claim, 'action': 'claim'}
            )

    def handle_player_respond(self, event):
        """Handle player's response to AI's claim"""
        for response, btn in self.response_buttons:
            if btn.handle_event(event):
                card = self.state.current_card
                claim = self.state.current_claim
                is_truth = (card == claim)

                # FIXED REWARD SYSTEM:
                # Correct guess = card goes to claimer (AI)
                # Wrong guess = card goes to responder (player)
                if (response == 'truth' and is_truth) or (response == 'bluff' and not is_truth):
                    # Player guessed CORRECTLY
                    self.state.ai_face_up.append(card)
                    self.message = f"Correct! The card was a {card}"
                    self.sub_message = "Card goes to AI"
                    reward = 1  # Player guessed correctly (good for player)
                else:
                    # Player guessed WRONG
                    self.state.player_face_up.append(card)
                    self.message = f"Wrong! The card was a {card}"
                    self.sub_message = "Card goes to you"
                    reward = -1  # Player guessed wrong (bad for player)

                # Check for terminal state
                winner = self.state.check_loss_condition()
                is_terminal = False

                if winner == 'player':
                    reward += 50  # Player wins
                    is_terminal = True
                elif winner == 'ai':
                    reward -= 50  # Player loses
                    is_terminal = True

                # Update agent's belief if available
                if self.agent and hasattr(self.agent, 'update_belief'):
                    cards_remaining = defaultdict(int)
                    for animal in GAME_CONFIG['animals']:
                        cards_remaining[animal] = 1
                    self.agent.update_belief(claim, card, cards_remaining)

                # Log state
                log_state = self.state.to_dict()
                log_state['game_over'] = is_terminal

                self.logger.log_state_action(
                    log_state,
                    {'player': 'player', 'response': response, 'action': 'respond'},
                    reward
                )

                self.state.current_card = None
                self.state.current_claim = None
                self.state.phase = 'claim'
                self.state.switch_claiming_player()

                # Check for game over
                if not self.check_game_over():
                    # Schedule next turn
                    if self.state.claiming_player == 'player' and not self.state.player_hand:
                        self.state.switch_claiming_player()

                    if self.state.claiming_player == 'ai' and self.state.ai_hand:
                        pygame.time.set_timer(pygame.USEREVENT + 2, 2000)

                return None
                self.state.phase = 'claim'
                self.state.switch_claiming_player()

                # Check for game over
                if not self.check_game_over():
                    # Schedule next turn
                    if self.state.claiming_player == 'player' and not self.state.player_hand:
                        self.state.switch_claiming_player()

                    if self.state.claiming_player == 'ai' and self.state.ai_hand:
                        pygame.time.set_timer(pygame.USEREVENT + 2, 2000)

                return

    def check_game_over(self) -> bool:
        """Check if game has ended"""
        winner = self.state.check_loss_condition()
        if winner:
            self.state.game_over = True
            self.state.winner = winner
            self.show_game_over()
            return True

        winner = self.state.check_empty_hands()
        if winner:
            self.state.game_over = True
            self.state.winner = winner
            self.show_game_over()
            return True

        return False

    def show_game_over(self):
        """Display game over screen"""
        self.logger.save_game()

        if self.state.winner == 'player':
            self.message = "🎉 YOU WIN! 🎉"
            self.message_color = GREEN
        elif self.state.winner == 'ai':
            self.message = "🤖 AI WINS! 🤖"
            self.message_color = RED
        else:
            self.message = "🤝 IT'S A TIE! 🤝"
            self.message_color = BLUE

        self.sub_message = "Press SPACE for new game or Q to quit"

    def run(self):
        """Main game loop"""
        running = True

        # Start first turn
        if self.state.claiming_player == 'ai':
            self.message = "AI will claim first!"
            pygame.time.set_timer(pygame.USEREVENT + 2, 1500)
        else:
            self.message = "Your turn! Select a card to claim"
            self.message_color = GREEN

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.USEREVENT + 1:
                    # AI response timer
                    pygame.time.set_timer(pygame.USEREVENT + 1, 0)
                    self.ai_respond()

                elif event.type == pygame.USEREVENT + 2:
                    # AI claim timer
                    pygame.time.set_timer(pygame.USEREVENT + 2, 0)
                    self.ai_claim()

                elif event.type == pygame.KEYDOWN:
                    if self.state.game_over:
                        if event.key == pygame.K_SPACE:
                            # New game
                            self.__init__(self.agent)
                            return self.run()
                        elif event.key == pygame.K_q:
                            running = False

                if not self.state.game_over:
                    # Handle game events
                    if self.state.phase == 'claim' and self.state.claiming_player == 'player':
                        self.handle_player_claim(event)

                    elif self.state.phase == 'respond' and self.state.claiming_player == 'ai':
                        self.handle_player_respond(event)

                    # Update button hover states
                    for _, btn in self.claim_buttons:
                        btn.handle_event(event)
                    for _, btn in self.response_buttons:
                        btn.handle_event(event)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main():
    """Entry point for the game"""
    print("\n" + "=" * 60)
    print("COCKROACH POKER - POMDP EDITION")
    print("=" * 60)
    print("\nOptions:")
    print("1. Play with RANDOM AI (data collection)")
    print("2. Play with TRAINED AI (after training)")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        print("\n" + "=" * 60)
        print("PLAYING WITH RANDOM AI")
        print("=" * 60)
        print("\nThe AI will make random guesses.")
        print("Play 20-30 games to collect training data.")
        print("After playing, run: python ai_agent.py (option 1)")
        print("\n" + "=" * 60)
        input("Press ENTER to start...")

        game = CockroachPokerUI()
        game.run()

    elif choice == '2':
        print("\n" + "=" * 60)
        print("PLAYING WITH TRAINED AI")
        print("=" * 60)

        try:
            from ai_agent import POMDPAgent

            print("\nInitializing AI agent...")
            agent = POMDPAgent(GAME_CONFIG)

            # Show AI statistics
            print("\nAI Statistics:")
            stats = agent.get_statistics()
            print(f"  States learned: {stats['states']}")
            print(f"  State-action pairs: {stats['state_action_pairs']}")
            if stats['state_action_pairs'] > 0:
                print(f"  Average Q-value: {stats['avg_q_value']:.3f}")
                print(f"  Observations: {stats['belief_state']['total_observations']}")

            if stats['states'] == 0:
                print("\n⚠️  WARNING: AI has no training data!")
                print("  The AI will make random guesses.")
                print("  Play some games (option 1), then train (python ai_agent.py)")
            else:
                print("\n✓ AI is trained and ready!")

            print("\n" + "=" * 60)
            print("\nGame Rules:")
            print("  • Select a card, then choose what to claim")
            print("  • When AI claims, decide: TRUTH or BLUFF")
            print("  • Get 3 of same animal → YOU LOSE")
            print("  • AI gets 3 of same animal → YOU WIN")
            print("\n" + "=" * 60)
            input("\nPress ENTER to start game...")

            game = CockroachPokerUI(ai_agent=agent)
            game.run()

            print("\n" + "=" * 60)
            print("Thanks for playing!")
            print("\nTo train AI on games you just played:")
            print("  python ai_agent.py")
            print("  Choose option 1 (Train agent)")
            print("=" * 60)

        except ImportError:
            print("\n❌ Error: ai_agent.py not found!")
            print("Make sure ai_agent.py is in the same directory.")

    elif choice == '3':
        print("Goodbye!")

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()