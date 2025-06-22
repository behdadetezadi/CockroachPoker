"""
Game UI using Pygame
"""

import pygame
import math
from typing import Optional, List, Tuple, Dict, Any
from ..game.game_manager import GameManager
from ..game.models import Card, GamePhase, PlayerType
from ..config import *


class Button:
    """Simple button class"""

    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: Tuple[int, int, int] = None, text_color: Tuple[int, int, int] = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color or COLORS['button_normal']
        self.text_color = text_color or COLORS['text_white']
        self.hovered = False
        self.enabled = True

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the button"""
        color = self.color
        if not self.enabled:
            color = COLORS['button_disabled']
        elif self.hovered:
            color = COLORS['button_hover']

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, COLORS['panel_border'], self.rect, 2)

        # Draw text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events, return True if clicked"""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.enabled and self.rect.collidepoint(event.pos):
                return True
        return False


class CardWidget:
    """Widget for displaying a card"""

    def __init__(self, x: int, y: int, card: Optional[Card] = None,
                 hidden: bool = False, selected: bool = False):
        self.rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        self.card = card
        self.hidden = hidden
        self.selected = selected
        self.hovered = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the card"""
        # Card background
        color = COLORS['card_bg']
        if self.selected:
            color = COLORS['card_selected']
        elif self.hovered:
            color = COLORS['card_hover']

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, COLORS['panel_border'], self.rect, 2)

        if self.hidden:
            # Draw card back
            font_large = pygame.font.Font(None, 48)
            text = font_large.render("🂠", True, COLORS['text_white'])
            text_rect = text.get_rect(center=self.rect.center)
            screen.blit(text, text_rect)
        elif self.card:
            # Draw creature emoji
            creature_emoji = CREATURE_EMOJIS.get(self.card.creature, "❓")

            # Draw emoji (larger font)
            font_large = pygame.font.Font(None, 36)
            emoji_surface = font_large.render(creature_emoji, True, COLORS['text_white'])
            emoji_rect = emoji_surface.get_rect(center=(self.rect.centerx, self.rect.centery - 20))
            screen.blit(emoji_surface, emoji_rect)

            # Draw creature name
            name_surface = font.render(self.card.creature, True, COLORS['text_white'])
            name_rect = name_surface.get_rect(center=(self.rect.centerx, self.rect.centery + 25))
            screen.blit(name_surface, name_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events, return True if clicked"""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Panel:
    """UI Panel for organizing content"""

    def __init__(self, x: int, y: int, width: int, height: int, title: str = ""):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw panel background and border"""
        pygame.draw.rect(screen, COLORS['panel_bg'], self.rect)
        pygame.draw.rect(screen, COLORS['panel_border'], self.rect, 2)

        if self.title:
            title_surface = font.render(self.title, True, COLORS['text_white'])
            screen.blit(title_surface, (self.rect.x + 10, self.rect.y + 5))


class GameUI:
    """Main game UI"""

    def __init__(self, screen: pygame.Surface, game_manager: GameManager):
        self.screen = screen
        self.game_manager = game_manager

        # Fonts
        pygame.font.init()
        self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)

        # UI state
        self.selected_card: Optional[Card] = None
        self.selected_claim: str = ""
        self.show_analytics = False
        self.show_face_detection = True

        # UI elements
        self.buttons: Dict[str, Button] = {}
        self.card_widgets: List[CardWidget] = []
        self.claim_buttons: List[Button] = []

        # Set up callbacks
        self.game_manager.set_state_change_callback(self._on_state_change)

        self._create_ui_elements()

    def _create_ui_elements(self):
        """Create UI buttons and elements"""
        # Main control buttons
        self.buttons['start_game'] = Button(50, 50, 120, 40, "Start Game")
        self.buttons['new_game'] = Button(180, 50, 120, 40, "New Game")
        self.buttons['analytics'] = Button(310, 50, 120, 40, "Analytics")
        self.buttons['face_detection'] = Button(440, 50, 140, 40, "Face Detection")

        # AI strategy buttons
        strategies = ['random', 'pattern', 'reinforcement', 'neural']
        for i, strategy in enumerate(strategies):
            self.buttons[f'ai_{strategy}'] = Button(
                600 + i * 120, 50, 110, 40,
                AI_STRATEGIES[strategy].replace(' AI', '').replace(' Agent', '')
            )

        # Challenge decision buttons
        self.buttons['challenge'] = Button(500, 400, 120, 50, "Challenge!")
        self.buttons['accept'] = Button(650, 400, 120, 50, "Accept")

        # Claim buttons (will be positioned dynamically)
        for creature in CREATURES:
            button = Button(0, 0, 80, 30, creature.title())
            self.claim_buttons.append(button)

        # Pass card button
        self.buttons['pass_card'] = Button(300, 600, 100, 40, "Pass Card")

    def _on_state_change(self):
        """Handle game state changes"""
        # Update UI elements based on game state
        pass

    def handle_event(self, event: pygame.event.Event):
        """Handle UI events"""
        game_state = self.game_manager.game_state

        # Handle main buttons
        if self.buttons['start_game'].handle_event(event):
            self.game_manager.start_new_game()

        if self.buttons['new_game'].handle_event(event):
            self.game_manager.start_new_game()

        if self.buttons['analytics'].handle_event(event):
            self.show_analytics = not self.show_analytics

        if self.buttons['face_detection'].handle_event(event):
            self.show_face_detection = not self.show_face_detection

        # Handle AI strategy selection
        for strategy in ['random', 'pattern', 'reinforcement', 'neural']:
            if self.buttons[f'ai_{strategy}'].handle_event(event):
                self.game_manager.set_ai_strategy(strategy)

        # Handle challenge decision buttons
        if (game_state.phase == GamePhase.CHALLENGE and
                game_state.pending_challenge and
                game_state.pending_challenge.receiver == PlayerType.HUMAN):

            if self.buttons['challenge'].handle_event(event):
                self.game_manager.human_challenge_decision(True)

            if self.buttons['accept'].handle_event(event):
                self.game_manager.human_challenge_decision(False)

        # Handle card selection for human player
        if (game_state.phase == GamePhase.PLAYING and
                game_state.current_player == PlayerType.HUMAN and
                not game_state.pending_challenge):

            for i, card in enumerate(game_state.human_player.hand):
                card_x = 100 + i * (CARD_WIDTH + CARD_MARGIN)
                card_y = 650
                card_rect = pygame.Rect(card_x, card_y, CARD_WIDTH, CARD_HEIGHT)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if card_rect.collidepoint(event.pos):
                        self.selected_card = card
                        self.selected_claim = ""  # Reset claim when selecting new card

            # Handle claim selection
            if self.selected_card:
                for i, creature in enumerate(CREATURES):
                    claim_x = 100 + i * 90
                    claim_y = 550
                    claim_rect = pygame.Rect(claim_x, claim_y, 80, 30)

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if claim_rect.collidepoint(event.pos):
                            self.selected_claim = creature

                # Handle pass card button
                if self.selected_claim and self.buttons['pass_card'].handle_event(event):
                    success = self.game_manager.human_pass_card(self.selected_card, self.selected_claim)
                    if success:
                        self.selected_card = None
                        self.selected_claim = ""

    def render(self):
        """Render the entire UI"""
        game_state = self.game_manager.game_state

        # Clear screen
        self.screen.fill(COLORS['background'])

        # Draw header
        self._draw_header()

        # Draw main game area based on phase
        if game_state.phase == GamePhase.SETUP:
            self._draw_setup_screen()
        elif game_state.phase in [GamePhase.PLAYING, GamePhase.CHALLENGE]:
            self._draw_game_screen()
        elif game_state.phase == GamePhase.GAME_OVER:
            self._draw_game_over_screen()

        # Draw side panels
        if self.show_analytics:
            self._draw_analytics_panel()

        if self.show_face_detection:
            self._draw_face_detection_panel()

        # Draw status message
        self._draw_status_message()

    def _draw_header(self):
        """Draw the header with title and controls"""
        # Title
        title_text = "🪳 Cockroach Poker - AI Bluff Detection Lab"
        title_surface = self.font_large.render(title_text, True, COLORS['text_white'])
        self.screen.blit(title_surface, (10, 10))

        # Control buttons
        for button in self.buttons.values():
            if button.text in ['Start Game', 'New Game', 'Analytics', 'Face Detection']:
                button.draw(self.screen, self.font_medium)

        # AI strategy buttons
        current_strategy = self.game_manager.ai_strategy.name if self.game_manager.ai_strategy else ""
        for strategy in ['random', 'pattern', 'reinforcement', 'neural']:
            button = self.buttons[f'ai_{strategy}']
            # Highlight current strategy
            if AI_STRATEGIES[strategy] in current_strategy:
                button.color = COLORS['success']
            else:
                button.color = COLORS['button_normal']
            button.draw(self.screen, self.font_small)

    def _draw_setup_screen(self):
        """Draw setup/welcome screen"""
        panel = Panel(200, 200, 800, 400, "Welcome to Cockroach Poker")
        panel.draw(self.screen, self.font_medium)

        instructions = [
            "This is an AI bluff detection research platform.",
            "Select an AI strategy and click 'Start Game' to begin.",
            "",
            "Game Rules:",
            "• Pass cards to your opponent, claiming what creature they are",
            "• You can lie about what creature you're passing!",
            "• Opponent decides whether to challenge your claim",
            "• Get 4 of the same creature and you lose",
            "• Play all your cards to win",
            "",
            "The AI analyzes your timing and facial expressions to detect bluffs!"
        ]

        y_offset = 250
        for line in instructions:
            if line:
                text_surface = self.font_small.render(line, True, COLORS['text_gray'])
                self.screen.blit(text_surface, (220, y_offset))
            y_offset += 25

    def _draw_game_screen(self):
        """Draw main game screen"""
        game_state = self.game_manager.game_state

        # Draw AI area (top)
        self._draw_ai_area()

        # Draw challenge area (middle)
        if game_state.phase == GamePhase.CHALLENGE:
            self._draw_challenge_area()

        # Draw human area (bottom)
        self._draw_human_area()

        # Draw game info
        self._draw_game_info()

    def _draw_ai_area(self):
        """Draw AI player area"""
        game_state = self.game_manager.game_state
        ai_player = game_state.ai_player

        # AI panel
        panel = Panel(50, 120, 600, 200,
                      f"AI Player ({self.game_manager.ai_strategy.name if self.game_manager.ai_strategy else 'None'})")
        panel.draw(self.screen, self.font_medium)

        # AI's cards on table
        cards_text = f"AI's Cards ({len(ai_player.cards_on_table)}):"
        text_surface = self.font_small.render(cards_text, True, COLORS['text_white'])
        self.screen.blit(text_surface, (70, 150))

        for i, card in enumerate(ai_player.cards_on_table):
            card_widget = CardWidget(70 + i * (CARD_WIDTH // 2 + 5), 170, card)
            card_widget.rect.width = CARD_WIDTH // 2
            card_widget.rect.height = CARD_HEIGHT // 2
            card_widget.draw(self.screen, self.font_small)

        # AI's hand (hidden)
        hand_text = f"AI's Hand ({len(ai_player.hand)} cards):"
        text_surface = self.font_small.render(hand_text, True, COLORS['text_white'])
        self.screen.blit(text_surface, (350, 150))

        for i in range(min(len(ai_player.hand), 8)):
            card_widget = CardWidget(350 + i * (CARD_WIDTH // 2 + 5), 170, hidden=True)
            card_widget.rect.width = CARD_WIDTH // 2
            card_widget.rect.height = CARD_HEIGHT // 2
            card_widget.draw(self.screen, self.font_small)

    def _draw_challenge_area(self):
        """Draw challenge decision area"""
        game_state = self.game_manager.game_state
        challenge = game_state.pending_challenge

        if not challenge:
            return

        # Challenge panel
        panel = Panel(200, 350, 800, 150, "Challenge Decision")
        panel.draw(self.screen, self.font_medium)

        if challenge.receiver == PlayerType.HUMAN:
            # Human needs to decide
            text = f"AI claims this card is a {challenge.claimed_creature.upper()}"
            text_surface = self.font_medium.render(text, True, COLORS['text_white'])
            text_rect = text_surface.get_rect(center=(600, 400))
            self.screen.blit(text_surface, text_rect)

            # Show challenge buttons
            self.buttons['challenge'].draw(self.screen, self.font_medium)
            self.buttons['accept'].draw(self.screen, self.font_medium)
        else:
            # AI is deciding
            text = f"AI is deciding whether to challenge your claim of '{challenge.claimed_creature}'"
            text_surface = self.font_medium.render(text, True, COLORS['text_white'])
            text_rect = text_surface.get_rect(center=(600, 425))
            self.screen.blit(text_surface, text_rect)

    def _draw_human_area(self):
        """Draw human player area"""
        game_state = self.game_manager.game_state
        human_player = game_state.human_player

        # Human's cards on table
        cards_text = f"Your Cards ({len(human_player.cards_on_table)}):"
        text_surface = self.font_small.render(cards_text, True, COLORS['text_white'])
        self.screen.blit(text_surface, (100, 500))

        for i, card in enumerate(human_player.cards_on_table):
            card_widget = CardWidget(100 + i * (CARD_WIDTH // 2 + 5), 520, card)
            card_widget.rect.width = CARD_WIDTH // 2
            card_widget.rect.height = CARD_HEIGHT // 2
            card_widget.draw(self.screen, self.font_small)

        # Human's hand
        if (game_state.phase == GamePhase.PLAYING and
                game_state.current_player == PlayerType.HUMAN and
                not game_state.pending_challenge):

            hand_text = f"Your Hand - Select a card to pass:"
            text_surface = self.font_small.render(hand_text, True, COLORS['text_white'])
            self.screen.blit(text_surface, (100, 620))

            # Draw hand cards
            for i, card in enumerate(human_player.hand):
                selected = (card == self.selected_card)
                card_widget = CardWidget(100 + i * (CARD_WIDTH + CARD_MARGIN), 650, card, selected=selected)
                card_widget.draw(self.screen, self.font_small)

            # Draw claim selection if card is selected
            if self.selected_card:
                claim_text = "Claim this card is a:"
                text_surface = self.font_small.render(claim_text, True, COLORS['text_white'])
                self.screen.blit(text_surface, (100, 580))

                for i, creature in enumerate(CREATURES):
                    x = 100 + i * 90
                    y = 600

                    # Button color based on selection
                    color = COLORS['success'] if creature == self.selected_claim else COLORS['button_normal']

                    button_rect = pygame.Rect(x, y, 80, 30)
                    pygame.draw.rect(self.screen, color, button_rect)
                    pygame.draw.rect(self.screen, COLORS['panel_border'], button_rect, 1)

                    text_surface = self.font_small.render(creature.title(), True, COLORS['text_white'])
                    text_rect = text_surface.get_rect(center=button_rect.center)
                    self.screen.blit(text_surface, text_rect)

                # Pass card button
                self.buttons['pass_card'].enabled = bool(self.selected_claim)
                self.buttons['pass_card'].draw(self.screen, self.font_small)

    def _draw_game_info(self):
        """Draw game information panel"""
        game_state = self.game_manager.game_state

        panel = Panel(700, 120, 300, 200, "Game Info")
        panel.draw(self.screen, self.font_medium)

        info_lines = [
            f"Round: {game_state.round_number}",
            f"Current Player: {game_state.current_player.value.title()}",
            f"Phase: {game_state.phase.value.title()}",
            "",
            f"Your Cards: {len(game_state.human_player.cards_on_table)}",
            f"AI Cards: {len(game_state.ai_player.cards_on_table)}",
            f"Your Hand: {len(game_state.human_player.hand)}",
            f"AI Hand: {len(game_state.ai_player.hand)}"
        ]

        y_offset = 150
        for line in info_lines:
            if line:
                text_surface = self.font_small.render(line, True, COLORS['text_white'])
                self.screen.blit(text_surface, (720, y_offset))
            y_offset += 20

    def _draw_game_over_screen(self):
        """Draw game over screen"""
        panel = Panel(300, 300, 600, 300, "Game Over")
        panel.draw(self.screen, self.font_large)

        # Show final statistics
        analytics = self.game_manager.get_analytics_summary()

        stats_lines = [
            f"Total Games: {analytics['total_games']}",
            f"Your Wins: {analytics['human_wins']}",
            f"AI Wins: {analytics['ai_wins']}",
            f"Win Rate: {analytics['win_rate']:.1f}%",
            "",
            f"Challenge Accuracy: {analytics['accuracy']:.1f}%",
            f"Bluff Detection Rate: {analytics['bluff_detection_rate']:.1f}%",
            f"Avg Decision Time: {analytics['avg_decision_time']:.1f}s"
        ]

        y_offset = 350
        for line in stats_lines:
            if line:
                text_surface = self.font_small.render(line, True, COLORS['text_white'])
                self.screen.blit(text_surface, (320, y_offset))
            y_offset += 25

    def _draw_analytics_panel(self):
        """Draw analytics side panel"""
        panel = Panel(1050, 120, 330, 500, "Live Analytics")
        panel.draw(self.screen, self.font_medium)

        analytics = self.game_manager.get_analytics_summary()

        metrics = [
            ("Games Played", f"{analytics['total_games']}"),
            ("Win Rate", f"{analytics['win_rate']:.1f}%"),
            ("Accuracy", f"{analytics['accuracy']:.1f}%"),
            ("Bluff Detection", f"{analytics['bluff_detection_rate']:.1f}%"),
            ("Avg Decision Time", f"{analytics['avg_decision_time']:.1f}s"),
            ("Total Decisions", f"{analytics['total_decisions']}"),
            ("AI Strategy", analytics['ai_strategy'])
        ]

        y_offset = 160
        for label, value in metrics:
            label_surface = self.font_small.render(f"{label}:", True, COLORS['text_gray'])
            value_surface = self.font_small.render(str(value), True, COLORS['text_white'])

            self.screen.blit(label_surface, (1070, y_offset))
            self.screen.blit(value_surface, (1200, y_offset))
            y_offset += 25

    def _draw_face_detection_panel(self):
        """Draw face detection panel"""
        panel = Panel(1050, 640, 330, 200, "Face Detection")
        panel.draw(self.screen, self.font_medium)

        expression = self.game_manager.get_current_facial_expression()
        emotion_analysis = self.game_manager.get_emotion_analysis()

        if expression:
            info_lines = [
                f"Emotion: {expression.emotion.title()}",
                f"Confidence: {expression.confidence:.2f}",
                f"Deception Likelihood: {emotion_analysis.get('deception_likelihood', 0):.2f}",
                "",
                "Status: Active" if self.game_manager.face_detector.is_detecting() else "Status: Inactive"
            ]
        else:
            info_lines = [
                "No expression detected",
                "",
                "Status: Inactive"
            ]

        y_offset = 680
        for line in info_lines:
            if line:
                color = COLORS['success'] if 'Active' in line else COLORS['text_white']
                text_surface = self.font_small.render(line, True, color)
                self.screen.blit(text_surface, (1070, y_offset))
            y_offset += 20

    def _draw_status_message(self):
        """Draw status message at bottom"""
        message = self.game_manager.game_state.last_action_message
        if message:
            text_surface = self.font_medium.render(message, True, COLORS['info'])
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            self.screen.blit(text_surface, text_rect)