import random
from collections import defaultdict
from typing import List, Optional, Tuple

import pygame

from core import DataLogger, GameState, calculate_cards_remaining, GAME_CONFIG
from agents.base import Agent

# --- pygame init must happen before font creation ---
pygame.init()

# UI constants
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
BACKGROUND = (45, 52, 64)
CARD_BG = (236, 239, 244)
MESSAGE_BG = (59, 66, 82)
TEXT_LIGHT = (236, 239, 244)

# Fonts
TITLE_FONT = pygame.font.Font(None, 48)
LARGE_FONT = pygame.font.Font(None, 36)
MEDIUM_FONT = pygame.font.Font(None, 28)
SMALL_FONT = pygame.font.Font(None, 22)

# Card dimensions
CARD_WIDTH = 100
CARD_HEIGHT = 140
CARD_SPACING = 15


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)
        surf = MEDIUM_FONT.render(self.text, True, WHITE)
        screen.blit(surf, surf.get_rect(center=self.rect.center))

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.current_color = (
                self.hover_color if self.rect.collidepoint(event.pos) else self.color
            )
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Card:
    def __init__(self, animal: str, x: int, y: int):
        self.animal = animal
        self.rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        self.hovered = False

    def draw(self, screen, face_up: bool = True, selected: bool = False):
        color = GOLD if selected else (GRAY if self.hovered else CARD_BG)
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=10)

        if face_up:
            text = MEDIUM_FONT.render(self.animal.capitalize(), True, BLACK)
            screen.blit(text, text.get_rect(center=self.rect.center))
        else:
            inner = self.rect.inflate(-10, -10)
            pygame.draw.rect(screen, PURPLE, inner, border_radius=6)
            for i in range(3):
                for j in range(4):
                    pygame.draw.circle(
                        screen, (180, 150, 255),
                        (inner.x + 15 + i * 25, inner.y + 20 + j * 30), 8
                    )
            back = LARGE_FONT.render('?', True, WHITE)
            screen.blit(back, back.get_rect(center=self.rect.center))

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class CockroachPokerUI:
    """Main game controller — UI + game flow. Accepts any Agent (or None for random AI)."""

    def __init__(self, ai_agent: Optional[Agent] = None):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Cockroach Poker - RL Edition")
        self.clock = pygame.time.Clock()

        self.state = GameState(GAME_CONFIG)
        self.logger = DataLogger()
        self.agent = ai_agent

        self.state.deal_hands()
        self.state.claiming_player = random.choice(['player', 'ai'])

        self.selected_card: Optional[int] = None
        self.selected_claim: Optional[str] = None
        self.message = ""
        self.message_color = BLACK
        self.sub_message = ""

        self.player_cards: List[Card] = []
        self.ai_cards: List[Card] = []
        self.claim_buttons: List[Tuple] = []
        self.response_buttons: List[Tuple] = []

        self._setup_ui()

    def _setup_ui(self):
        bw, bh = 120, 40
        animals = GAME_CONFIG['animals']

        # First row (up to 4 animals)
        row1 = animals[:4]
        start_x = (SCREEN_WIDTH - len(row1) * (bw + 10)) // 2
        for i, animal in enumerate(row1):
            x = start_x + i * (bw + 10)
            self.claim_buttons.append(
                (animal, Button(x, SCREEN_HEIGHT // 2 - 40, bw, bh,
                                animal.capitalize(), BLUE, DARK_GREEN))
            )

        # Second row (animals 5+ if config ever expands)
        row2 = animals[4:]
        if row2:
            start_x = (SCREEN_WIDTH - len(row2) * (bw + 10)) // 2
            for i, animal in enumerate(row2):
                x = start_x + i * (bw + 10)
                self.claim_buttons.append(
                    (animal, Button(x, SCREEN_HEIGHT // 2 + 10, bw, bh,
                                    animal.capitalize(), BLUE, DARK_GREEN))
                )

        self.response_buttons = [
            ('truth', Button(SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 - 20,
                             120, 60, "TRUTH", GREEN, DARK_GREEN)),
            ('bluff', Button(SCREEN_WIDTH // 2 + 10, SCREEN_HEIGHT // 2 - 20,
                             120, 60, "BLUFF", RED, DARK_GRAY)),
        ]

    def _cards_remaining(self):
        return calculate_cards_remaining(self.state.to_dict(), GAME_CONFIG)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _update_card_positions(self):
        self.player_cards = []
        total = len(self.state.player_hand) * (CARD_WIDTH + CARD_SPACING) - CARD_SPACING
        sx = (SCREEN_WIDTH - total) // 2
        y = SCREEN_HEIGHT - CARD_HEIGHT - 30
        for i, animal in enumerate(self.state.player_hand):
            self.player_cards.append(Card(animal, sx + i * (CARD_WIDTH + CARD_SPACING), y))

        self.ai_cards = []
        total = len(self.state.ai_hand) * (CARD_WIDTH + CARD_SPACING) - CARD_SPACING
        sx = (SCREEN_WIDTH - total) // 2
        for i in range(len(self.state.ai_hand)):
            self.ai_cards.append(Card('unknown', sx + i * (CARD_WIDTH + CARD_SPACING), 90))

    def _draw_face_up_cards(self):
        if self.state.player_face_up:
            y = SCREEN_HEIGHT - 230
            panel = pygame.Rect(10, y - 40, 600, 80)
            pygame.draw.rect(self.screen, MESSAGE_BG, panel, border_radius=10)
            pygame.draw.rect(self.screen, TEXT_LIGHT, panel, 2, border_radius=10)
            self.screen.blit(MEDIUM_FONT.render("Your Penalty Cards:", True, TEXT_LIGHT), (20, y - 35))
            counts: Dict[str, int] = defaultdict(int)
            for card in self.state.player_face_up:
                counts[card] += 1
            x = 20
            for animal, count in sorted(counts.items()):
                if count >= GAME_CONFIG['lose_threshold']:
                    pygame.draw.rect(self.screen, RED, (x - 5, y - 5, 110, 30), 3, border_radius=5)
                self.screen.blit(SMALL_FONT.render(f"{animal}: {count}", True, TEXT_LIGHT), (x, y))
                x += 120

        if self.state.ai_face_up:
            y = 200
            panel = pygame.Rect(10, y - 40, 600, 80)
            pygame.draw.rect(self.screen, MESSAGE_BG, panel, border_radius=10)
            pygame.draw.rect(self.screen, TEXT_LIGHT, panel, 2, border_radius=10)
            self.screen.blit(MEDIUM_FONT.render("AI Penalty Cards:", True, TEXT_LIGHT), (20, y - 35))
            counts = defaultdict(int)
            for card in self.state.ai_face_up:
                counts[card] += 1
            x = 20
            for animal, count in sorted(counts.items()):
                if count >= GAME_CONFIG['lose_threshold']:
                    pygame.draw.rect(self.screen, RED, (x - 5, y - 5, 110, 30), 3, border_radius=5)
                self.screen.blit(SMALL_FONT.render(f"{animal}: {count}", True, TEXT_LIGHT), (x, y))
                x += 120

    def draw(self):
        self.screen.fill(BACKGROUND)

        # Title bar
        pygame.draw.rect(self.screen, MESSAGE_BG, pygame.Rect(0, 0, SCREEN_WIDTH, 80))
        pygame.draw.rect(self.screen, PURPLE, (0, 78, SCREEN_WIDTH, 2))
        title = TITLE_FONT.render("Cockroach Poker", True, PURPLE)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 40)))

        # Message area
        if self.message:
            msg_y = SCREEN_HEIGHT // 2 - 100
            surf = LARGE_FONT.render(self.message, True, TEXT_LIGHT)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, msg_y))
            bg = rect.inflate(60, 30)
            pygame.draw.rect(self.screen, MESSAGE_BG, bg, border_radius=12)
            pygame.draw.rect(self.screen, self.message_color, bg, 3, border_radius=12)
            self.screen.blit(surf, rect)
            if self.sub_message:
                sub = MEDIUM_FONT.render(self.sub_message, True, TEXT_LIGHT)
                self.screen.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, msg_y + 40)))

        self._update_card_positions()
        for card in self.ai_cards:
            card.draw(self.screen, face_up=False)
        for i, card in enumerate(self.player_cards):
            card.draw(self.screen, face_up=True, selected=(self.selected_card == i))

        self._draw_face_up_cards()

        # Claim buttons (shown after card selected, waiting for claim choice)
        if (self.state.phase == 'claim'
                and self.state.claiming_player == 'player'
                and self.selected_card is not None
                and self.selected_claim is None):
            label = LARGE_FONT.render("Choose what to claim:", True, TEXT_LIGHT)
            label_rect = label.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 140))
            bg = label_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, MESSAGE_BG, bg, border_radius=10)
            pygame.draw.rect(self.screen, GOLD, bg, 2, border_radius=10)
            self.screen.blit(label, label_rect)
            for _, btn in self.claim_buttons:
                btn.draw(self.screen)

        # Response buttons (shown when AI has claimed)
        if self.state.phase == 'respond' and self.state.claiming_player == 'ai':
            for _, btn in self.response_buttons:
                btn.draw(self.screen)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def handle_player_claim(self, event):
        if self.selected_card is None:
            for i, card in enumerate(self.player_cards):
                if card.handle_event(event):
                    self.selected_card = i
                    self.message = "Card selected! Now choose what to claim:"
                    self.message_color = BLUE
                    return
        else:
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
                        {'player': 'player', 'card': card, 'claim': animal, 'action': 'claim'},
                    )
                    self.selected_card = None
                    self.selected_claim = None
                    pygame.time.set_timer(pygame.USEREVENT + 1, 1500)
                    return

    def ai_respond(self):
        """AI responds to the player's claim — no peeking at current_card."""
        observable = {
            'ai_hand': self.state.ai_hand,
            'player_face_up': self.state.player_face_up,
            'ai_face_up': self.state.ai_face_up,
            'current_claim': self.state.current_claim,
        }

        response = (
            self.agent.choose_response(observable)
            if self.agent
            else random.choice(['truth', 'bluff'])
        )

        card = self.state.current_card
        claim = self.state.current_claim
        is_truth = (card == claim)

        if (response == 'truth' and is_truth) or (response == 'bluff' and not is_truth):
            self.state.player_face_up.append(card)
            self.message = f"AI calls {response.upper()}! Correct! The card was a {card}"
            self.sub_message = "Card goes to you"
            reward = 1
        else:
            self.state.ai_face_up.append(card)
            self.message = f"AI calls {response.upper()}! Wrong! The card was a {card}"
            self.sub_message = "Card goes to AI"
            reward = -1

        winner = self.state.check_loss_condition() or self.state.check_empty_hands()
        is_terminal = winner is not None
        if winner == 'ai':
            reward += 50
        elif winner == 'player':
            reward -= 50

        if self.agent:
            self.agent.update_belief(claim, card, self._cards_remaining())

        log_state = {
            'ai_hand': self.state.ai_hand.copy(),
            'player_face_up': self.state.player_face_up.copy(),
            'ai_face_up': self.state.ai_face_up.copy(),
            'current_claim': claim,
            'current_card': card,
            'game_over': is_terminal,
        }
        self.logger.log_state_action(
            log_state,
            {'player': 'ai', 'response': response, 'action': 'respond'},
            reward,
        )

        self.state.current_card = None
        self.state.current_claim = None
        self.state.phase = 'claim'
        self.state.switch_claiming_player()

        if not self._check_game_over():
            if self.state.claiming_player == 'ai' and self.state.ai_hand:
                pygame.time.set_timer(pygame.USEREVENT + 2, 2000)

    def ai_claim(self):
        """AI picks a card from its hand and makes a claim."""
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
                {'player': 'ai', 'claim': claim, 'action': 'claim'},
            )

    def handle_player_respond(self, event):
        """Player responds truth/bluff to the AI's claim."""
        for response, btn in self.response_buttons:
            if btn.handle_event(event):
                card = self.state.current_card
                claim = self.state.current_claim
                is_truth = (card == claim)

                if (response == 'truth' and is_truth) or (response == 'bluff' and not is_truth):
                    self.state.ai_face_up.append(card)
                    self.message = f"Correct! The card was a {card}"
                    self.sub_message = "Card goes to AI"
                    reward = 1
                else:
                    self.state.player_face_up.append(card)
                    self.message = f"Wrong! The card was a {card}"
                    self.sub_message = "Card goes to you"
                    reward = -1

                winner = self.state.check_loss_condition() or self.state.check_empty_hands()
                is_terminal = winner is not None
                if winner == 'player':
                    reward += 50
                elif winner == 'ai':
                    reward -= 50

                if self.agent:
                    self.agent.update_belief(claim, card, self._cards_remaining())

                log_state = self.state.to_dict()
                log_state['game_over'] = is_terminal
                self.logger.log_state_action(
                    log_state,
                    {'player': 'player', 'response': response, 'action': 'respond'},
                    reward,
                )

                self.state.current_card = None
                self.state.current_claim = None
                self.state.phase = 'claim'
                self.state.switch_claiming_player()

                if not self._check_game_over():
                    if self.state.claiming_player == 'player' and not self.state.player_hand:
                        self.state.switch_claiming_player()
                    if self.state.claiming_player == 'ai' and self.state.ai_hand:
                        pygame.time.set_timer(pygame.USEREVENT + 2, 2000)
                return

    # ------------------------------------------------------------------
    # Game over
    # ------------------------------------------------------------------

    def _check_game_over(self) -> bool:
        winner = self.state.check_loss_condition() or self.state.check_empty_hands()
        if winner:
            self.state.game_over = True
            self.state.winner = winner
            self._show_game_over()
            return True
        return False

    def _show_game_over(self):
        self.logger.save_game()
        if self.state.winner == 'player':
            self.message, self.message_color = "YOU WIN!", GREEN
        elif self.state.winner == 'ai':
            self.message, self.message_color = "AI WINS!", RED
        else:
            self.message, self.message_color = "IT'S A TIE!", BLUE
        self.sub_message = "Press SPACE for new game or Q to quit"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        if self.state.claiming_player == 'ai':
            self.message = "AI will claim first!"
            pygame.time.set_timer(pygame.USEREVENT + 2, 1500)
        else:
            self.message = "Your turn! Select a card to claim"
            self.message_color = GREEN

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.USEREVENT + 1:
                    pygame.time.set_timer(pygame.USEREVENT + 1, 0)
                    self.ai_respond()

                elif event.type == pygame.USEREVENT + 2:
                    pygame.time.set_timer(pygame.USEREVENT + 2, 0)
                    self.ai_claim()

                elif event.type == pygame.KEYDOWN and self.state.game_over:
                    if event.key == pygame.K_SPACE:
                        self.__init__(self.agent)
                        return self.run()
                    elif event.key == pygame.K_q:
                        running = False

                if not self.state.game_over:
                    if self.state.phase == 'claim' and self.state.claiming_player == 'player':
                        self.handle_player_claim(event)
                    elif self.state.phase == 'respond' and self.state.claiming_player == 'ai':
                        self.handle_player_respond(event)

                    for _, btn in self.claim_buttons:
                        btn.handle_event(event)
                    for _, btn in self.response_buttons:
                        btn.handle_event(event)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
