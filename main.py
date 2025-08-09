#!/usr/bin/env python3
"""
Cockroach Poker - Simplified Version
Base game with random AI strategy only
"""

import pygame
import sys
import random
import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
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

# Game constants
CREATURES = ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
CARDS_PER_CREATURE = 6
STARTING_HAND_SIZE = 5
LOSE_CONDITION = 3


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


class RandomAI:
    """Simple random AI strategy"""

    def __init__(self):
        self.name = "Random AI"
        self.description = "Makes completely random decisions"

    def should_call_bluff(self, claimed_creature: str, game_state: Dict) -> bool:
        """50/50 random decision"""
        return random.random() < 0.5

    def choose_claim(self, actual_creature: str) -> str:
        """Random claim - 50% truth, 50% lie"""
        if random.random() < 0.5:
            return actual_creature
        else:
            possible_lies = [c for c in CREATURES if c != actual_creature]
            return random.choice(possible_lies)


class CockroachPokerGame:
    """Main game class"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Cockroach Poker")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.big_font = pygame.font.Font(None, 36)

        # Game state
        self.phase = GamePhase.MENU
        self.human_player = Player("Human", [], [])
        self.ai_player = Player("AI", [], [])
        self.current_card = None
        self.claimed_creature = None
        self.message = "Welcome to Cockroach Poker!"
        self.message_timer = 0

        # AI
        self.ai_strategy = RandomAI()

        # UI state
        self.selected_card_index = -1
        self.selected_claim = ""

        # Stats
        self.game_count = 0
        self.human_wins = 0
        self.ai_wins = 0

    def init_new_game(self):
        """Start a new game"""
        # Create deck
        deck = []
        for creature in CREATURES:
            for _ in range(CARDS_PER_CREATURE):
                deck.append(Card(creature))

        random.shuffle(deck)

        # Deal hands
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

        self.set_message(
            "Your turn! Select a card from your hand, then choose what creature to claim it is. You can lie or tell the truth!")

    def set_message(self, text: str, duration: float = 6.0):
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

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)

            # Handle timer events properly
            elif event.type == pygame.USEREVENT + 1:
                self.ai_challenge_decision()
                pygame.time.set_timer(pygame.USEREVENT + 1, 0)  # Cancel timer
            elif event.type == pygame.USEREVENT + 2:
                self.ai_turn()
                pygame.time.set_timer(pygame.USEREVENT + 2, 0)  # Cancel timer

        return True

    def handle_mouse_click(self, pos):
        """Handle mouse clicks"""
        if self.phase == GamePhase.HUMAN_TURN:
            self._handle_human_turn_click(pos)
        elif self.phase == GamePhase.HUMAN_CHALLENGE:
            self._handle_human_challenge_click(pos)

    def _handle_human_turn_click(self, pos):
        """Handle clicks during human turn"""
        # Check card selection
        card_y = WINDOW_HEIGHT - 140
        for i, card in enumerate(self.human_player.hand):
            card_x = 50 + i * 120
            card_rect = pygame.Rect(card_x, card_y, 100, 120)
            if card_rect.collidepoint(pos):
                self.selected_card_index = i
                self.selected_claim = ""
                return

        # Check creature claim buttons
        if self.selected_card_index >= 0:
            claims_y = WINDOW_HEIGHT - 220
            for i, creature in enumerate(CREATURES):
                claim_x = 50 + i * 150
                claim_rect = pygame.Rect(claim_x, claims_y, 140, 35)
                if claim_rect.collidepoint(pos):
                    self.selected_claim = creature
                    return

            # Check pass button
            pass_rect = pygame.Rect(50, WINDOW_HEIGHT - 180, 120, 35)
            if pass_rect.collidepoint(pos) and self.selected_claim:
                self.human_pass_card()

    def _handle_human_challenge_click(self, pos):
        """Handle clicks during challenge phase"""
        challenge_rect = pygame.Rect(WINDOW_WIDTH // 2 - 140, WINDOW_HEIGHT // 2 + 80, 120, 50)
        accept_rect = pygame.Rect(WINDOW_WIDTH // 2 + 20, WINDOW_HEIGHT // 2 + 80, 120, 50)

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

        self.set_message(
            f"You passed a '{self.claimed_creature}' to the AI. The AI is thinking about whether to challenge you...")
        self.phase = GamePhase.AI_CHALLENGE

        # AI decides after a delay
        pygame.time.set_timer(pygame.USEREVENT + 1, random.randint(1500, 3000))

        self.selected_card_index = -1
        self.selected_claim = ""

    def ai_challenge_decision(self):
        """AI decides whether to challenge"""
        if self.phase != GamePhase.AI_CHALLENGE:
            return

        # Simple game state for AI
        game_state = {
            'human_creature_counts': {card.creature: self.human_player.get_creature_count(card.creature)
                                      for card in self.human_player.table_cards},
            'ai_creature_counts': {card.creature: self.ai_player.get_creature_count(card.creature)
                                   for card in self.ai_player.table_cards}
        }

        ai_calls_bluff = self.ai_strategy.should_call_bluff(self.claimed_creature, game_state)
        is_bluff = self.current_card.creature != self.claimed_creature

        # Resolve challenge
        self._resolve_challenge(ai_calls_bluff, is_bluff, "AI")

        # Check win conditions
        if self.check_game_over():
            return

        # AI's turn next
        self.phase = GamePhase.AI_TURN
        pygame.time.set_timer(pygame.USEREVENT + 2, 2000)

    def ai_turn(self):
        """AI takes its turn"""
        if self.phase != GamePhase.AI_TURN or not self.ai_player.hand:
            return

        card = random.choice(self.ai_player.hand)
        self.ai_player.hand.remove(card)

        claimed = self.ai_strategy.choose_claim(card.creature)

        self.current_card = card
        self.claimed_creature = claimed

        self.set_message(
            f"AI passes you a card and claims it's a '{claimed}'. Look at your hand and decide: Is the AI telling the truth or bluffing?")
        self.phase = GamePhase.HUMAN_CHALLENGE

    def human_challenge_decision(self, challenges: bool):
        """Human makes challenge decision"""
        if self.phase != GamePhase.HUMAN_CHALLENGE:
            return

        is_bluff = self.current_card.creature != self.claimed_creature

        # Resolve challenge
        self._resolve_challenge(challenges, is_bluff, "Human")

        # Check win conditions
        if self.check_game_over():
            return

        # Continue game
        if self.human_player.hand:
            self.phase = GamePhase.HUMAN_TURN
            self.set_message("Your turn again! Select a card from your hand and choose what creature to claim it is.")
        elif self.ai_player.hand:
            self.phase = GamePhase.AI_TURN
            pygame.time.set_timer(pygame.USEREVENT + 2, 1000)
        else:
            self.phase = GamePhase.GAME_OVER
            self.set_message("All cards played!")

    def _resolve_challenge(self, challenger_calls_bluff: bool, is_bluff: bool, challenger: str):
        """Resolve challenge using Cockroach Poker rules"""
        actual_creature = self.current_card.creature
        claimed_creature = self.claimed_creature

        if challenger_calls_bluff:
            # Challenger says "This is a BLUFF!"
            if is_bluff:
                # Challenger was CORRECT - claimer gets the card
                if challenger == "AI":
                    self.human_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"AI called BLUFF and was RIGHT! The card was actually a {actual_creature}, not a {claimed_creature}. You (the liar) get the {actual_creature}.")
                else:
                    self.ai_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"You called BLUFF and were RIGHT! The card was actually a {actual_creature}, not a {claimed_creature}. AI (the liar) gets the {actual_creature}.")
            else:
                # Challenger was WRONG - challenger gets the card
                if challenger == "AI":
                    self.ai_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"AI called BLUFF but was WRONG! The card really was a {actual_creature} as claimed. AI (wrong challenger) gets the {actual_creature}.")
                else:
                    self.human_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"You called BLUFF but were WRONG! The card really was a {actual_creature} as claimed. You (wrong challenger) get the {actual_creature}.")
        else:
            # Challenger says "This is TRUE!"
            if is_bluff:
                # Challenger was WRONG - challenger gets the card
                if challenger == "AI":
                    self.ai_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"AI called TRUE but was WRONG! The card was actually a {actual_creature}, not a {claimed_creature}. AI (wrong challenger) gets the {actual_creature}.")
                else:
                    self.human_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"You called TRUE but were WRONG! The card was actually a {actual_creature}, not a {claimed_creature}. You (wrong challenger) get the {actual_creature}.")
            else:
                # Challenger was CORRECT - claimer gets the card
                if challenger == "AI":
                    self.human_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"AI called TRUE and was RIGHT! The card really was a {actual_creature} as claimed. You (the claimer) get the {actual_creature}.")
                else:
                    self.ai_player.table_cards.append(self.current_card)
                    self.set_message(
                        f"You called TRUE and were RIGHT! The card really was a {actual_creature} as claimed. AI (the claimer) gets the {actual_creature}.")

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
        title = self.big_font.render("Cockroach Poker", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)

        instruction = self.font.render("Press SPACE to start", True, WHITE)
        inst_rect = instruction.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(instruction, inst_rect)

        if self.game_count > 0:
            stats = self.font.render(
                f"Games: {self.game_count} | Your Wins: {self.human_wins} | AI Wins: {self.ai_wins}",
                True, LIGHT_GRAY
            )
            stats_rect = stats.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(stats, stats_rect)

    def draw_game(self):
        """Draw game screen"""
        # Phase indicator
        phase_names = {
            GamePhase.HUMAN_TURN: "YOUR TURN",
            GamePhase.AI_TURN: "AI TURN",
            GamePhase.HUMAN_CHALLENGE: "YOUR DECISION",
            GamePhase.AI_CHALLENGE: "AI DECIDING",
            GamePhase.GAME_OVER: "GAME OVER"
        }

        phase_text = phase_names.get(self.phase, "UNKNOWN")
        phase_surface = self.big_font.render(phase_text, True, YELLOW)
        phase_rect = phase_surface.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(phase_surface, phase_rect)

        # AI area
        self.draw_ai_area()

        # Human area
        self.draw_human_area()

        # Challenge interface
        if self.phase == GamePhase.HUMAN_CHALLENGE:
            self.draw_challenge_interface()

        # Message display
        if self.message and self.message_timer > 0:
            msg_surface = self.font.render(self.message, True, WHITE)
            msg_rect = msg_surface.get_rect(center=(WINDOW_WIDTH // 2, 80))

            # Background
            bg_rect = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, BLACK, bg_rect)
            pygame.draw.rect(self.screen, WHITE, bg_rect, 1)

            self.screen.blit(msg_surface, msg_rect)

        # Controls help
        if self.phase == GamePhase.GAME_OVER:
            restart_text = self.font.render("Press R to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            self.screen.blit(restart_text, restart_rect)

    def draw_ai_area(self):
        """Draw AI information area"""
        # AI cards on table
        ai_text = self.font.render(f"AI Cards on Table ({len(self.ai_player.table_cards)}):", True, WHITE)
        self.screen.blit(ai_text, (50, 120))

        for i, card in enumerate(self.ai_player.table_cards):
            card_rect = pygame.Rect(50 + i * 70, 150, 60, 80)
            pygame.draw.rect(self.screen, WHITE, card_rect)
            pygame.draw.rect(self.screen, BLACK, card_rect, 2)

            creature_text = self.small_font.render(card.creature[:4], True, BLACK)
            text_rect = creature_text.get_rect(center=card_rect.center)
            self.screen.blit(creature_text, text_rect)

        # AI creature counts
        ai_counts = {}
        for card in self.ai_player.table_cards:
            ai_counts[card.creature] = ai_counts.get(card.creature, 0) + 1

        count_x = 500
        count_y = 120
        for creature, count in ai_counts.items():
            color = RED if count >= LOSE_CONDITION else YELLOW if count >= 2 else WHITE
            count_text = self.font.render(f"{creature}: {count}", True, color)
            self.screen.blit(count_text, (count_x, count_y))
            count_y += 25

        # AI hand
        ai_hand_text = self.font.render(f"AI Hand: {len(self.ai_player.hand)} cards", True, WHITE)
        self.screen.blit(ai_hand_text, (50, 250))

    def draw_human_area(self):
        """Draw human player area"""
        # Human table cards
        human_text = self.font.render(f"Your Cards on Table ({len(self.human_player.table_cards)}):", True, WHITE)
        self.screen.blit(human_text, (50, 300))

        for i, card in enumerate(self.human_player.table_cards):
            card_rect = pygame.Rect(50 + i * 70, 330, 60, 80)
            pygame.draw.rect(self.screen, WHITE, card_rect)
            pygame.draw.rect(self.screen, BLACK, card_rect, 2)

            creature_text = self.small_font.render(card.creature[:4], True, BLACK)
            text_rect = creature_text.get_rect(center=card_rect.center)
            self.screen.blit(creature_text, text_rect)

        # Human creature counts
        human_counts = {}
        for card in self.human_player.table_cards:
            human_counts[card.creature] = human_counts.get(card.creature, 0) + 1

        count_x = 500
        count_y = 300
        for creature, count in human_counts.items():
            color = RED if count >= LOSE_CONDITION else YELLOW if count >= 2 else WHITE
            count_text = self.font.render(f"{creature}: {count}", True, color)
            self.screen.blit(count_text, (count_x, count_y))
            count_y += 25

        # Show human hand during their turn OR during challenge phase
        if self.phase == GamePhase.HUMAN_TURN:
            self.draw_human_hand_interface()
        elif self.phase == GamePhase.HUMAN_CHALLENGE:
            self.draw_human_hand_display()

    def draw_human_hand_display(self):
        """Draw human hand for viewing during challenge (non-interactive)"""
        hand_text = self.font.render("Your Hand:", True, LIGHT_GRAY)
        self.screen.blit(hand_text, (50, WINDOW_HEIGHT - 200))

        for i, card in enumerate(self.human_player.hand):
            card_x = 50 + i * 120
            card_y = WINDOW_HEIGHT - 160
            card_rect = pygame.Rect(card_x, card_y, 100, 120)

            pygame.draw.rect(self.screen, WHITE, card_rect)
            pygame.draw.rect(self.screen, GRAY, card_rect, 2)

            creature_text = self.small_font.render(card.creature, True, BLACK)
            text_rect = creature_text.get_rect(center=card_rect.center)
            self.screen.blit(creature_text, text_rect)

    def draw_human_hand_interface(self):
        """Draw human hand and claiming interface"""
        hand_text = self.font.render("Your Hand - Click to select:", True, YELLOW)
        self.screen.blit(hand_text, (50, WINDOW_HEIGHT - 260))

        for i, card in enumerate(self.human_player.hand):
            card_x = 50 + i * 120
            card_y = WINDOW_HEIGHT - 140
            card_rect = pygame.Rect(card_x, card_y, 100, 120)

            color = YELLOW if i == self.selected_card_index else WHITE
            pygame.draw.rect(self.screen, color, card_rect)
            pygame.draw.rect(self.screen, BLACK, card_rect, 2)

            creature_text = self.small_font.render(card.creature, True, BLACK)
            text_rect = creature_text.get_rect(center=card_rect.center)
            self.screen.blit(creature_text, text_rect)

        # Claim buttons
        if self.selected_card_index >= 0:
            claim_text = self.font.render("Claim this card is a:", True, WHITE)
            self.screen.blit(claim_text, (50, WINDOW_HEIGHT - 260))

            for i, creature in enumerate(CREATURES):
                claim_x = 50 + i * 150
                claim_y = WINDOW_HEIGHT - 220
                claim_rect = pygame.Rect(claim_x, claim_y, 140, 35)

                color = YELLOW if creature == self.selected_claim else LIGHT_GRAY
                pygame.draw.rect(self.screen, color, claim_rect)
                pygame.draw.rect(self.screen, BLACK, claim_rect, 2)

                text = self.small_font.render(creature, True, BLACK)
                text_rect = text.get_rect(center=claim_rect.center)
                self.screen.blit(text, text_rect)

            # Pass button
            if self.selected_claim:
                pass_rect = pygame.Rect(50, WINDOW_HEIGHT - 180, 120, 35)
                pygame.draw.rect(self.screen, BLUE, pass_rect)
                pygame.draw.rect(self.screen, BLACK, pass_rect, 2)

                pass_text = self.font.render("PASS CARD", True, WHITE)
                text_rect = pass_text.get_rect(center=pass_rect.center)
                self.screen.blit(pass_text, text_rect)

    def draw_challenge_interface(self):
        """Draw challenge decision interface"""
        challenge_text = self.big_font.render(f"AI claims: {self.claimed_creature}", True, WHITE)
        text_rect = challenge_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
        self.screen.blit(challenge_text, text_rect)

        instruction1 = self.font.render("Is the AI telling the truth or bluffing?", True, WHITE)
        inst1_rect = instruction1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 10))
        self.screen.blit(instruction1, inst1_rect)

        instruction2 = self.small_font.render("Remember: Whoever is wrong gets the card!", True, YELLOW)
        inst2_rect = instruction2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 15))
        self.screen.blit(instruction2, inst2_rect)

        # Buttons
        challenge_rect = pygame.Rect(WINDOW_WIDTH // 2 - 140, WINDOW_HEIGHT // 2 + 50, 120, 50)
        accept_rect = pygame.Rect(WINDOW_WIDTH // 2 + 20, WINDOW_HEIGHT // 2 + 50, 120, 50)

        pygame.draw.rect(self.screen, RED, challenge_rect)
        pygame.draw.rect(self.screen, BLUE, accept_rect)
        pygame.draw.rect(self.screen, BLACK, challenge_rect, 2)
        pygame.draw.rect(self.screen, BLACK, accept_rect, 2)

        challenge_text = self.font.render("BLUFF!", True, WHITE)
        accept_text = self.font.render("TRUE!", True, WHITE)

        # Add explanatory text below buttons
        bluff_explain = self.small_font.render("(AI is lying)", True, WHITE)
        true_explain = self.small_font.render("(AI is honest)", True, WHITE)

        chal_rect = challenge_text.get_rect(center=(challenge_rect.centerx, challenge_rect.centery - 5))
        acc_rect = accept_text.get_rect(center=(accept_rect.centerx, accept_rect.centery - 5))

        bluff_exp_rect = bluff_explain.get_rect(center=(challenge_rect.centerx, challenge_rect.centery + 15))
        true_exp_rect = true_explain.get_rect(center=(accept_rect.centerx, accept_rect.centery + 15))

        self.screen.blit(challenge_text, chal_rect)
        self.screen.blit(accept_text, acc_rect)
        self.screen.blit(bluff_explain, bluff_exp_rect)
        self.screen.blit(true_explain, true_exp_rect)

    def run(self):
        """Main game loop"""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            running = self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    try:
        game = CockroachPokerGame()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        pygame.quit()
        sys.exit()