#!/usr/bin/env python3
"""
Cockroach Poker - AI Bluff Detection Simulator
Main entry point for the game
"""

import pygame
import sys
from src.game.game_manager import GameManager
from src.ui.game_ui import GameUI
from src.config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, WINDOW_TITLE


def main():
    """Main game loop"""
    # Initialize Pygame
    pygame.init()

    # Create window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    # Initialize game components
    game_manager = GameManager()
    game_ui = GameUI(screen, game_manager)

    # Game loop
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                game_ui.handle_event(event)

        # Update game state
        game_manager.update(dt)

        # Render
        screen.fill((20, 40, 20))  # Dark green background
        game_ui.render()
        pygame.display.flip()

    # Cleanup
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()