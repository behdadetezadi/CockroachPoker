import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

GAME_CONFIG = {
    'hand_size': 5,
    'animals': ['fly', 'rat', 'toad', 'bat'],
    'lose_threshold': 3,
    'cards_per_animal': 8,
}


def calculate_cards_remaining(state: Dict, config: Dict) -> Dict[str, int]:
    """
    Cards not yet accounted for from the AI's perspective.
    Subtracts AI's own hand and both face-up piles from the total deck.
    The remainder is the pool that could be in the player's hand or deck.
    """
    cards_remaining: Dict[str, int] = defaultdict(int)
    for animal in config['animals']:
        cards_remaining[animal] = config['cards_per_animal']
    for card in state.get('ai_hand', []):
        cards_remaining[card] -= 1
    for card in state.get('player_face_up', []):
        cards_remaining[card] -= 1
    for card in state.get('ai_face_up', []):
        cards_remaining[card] -= 1
    for animal in config['animals']:
        cards_remaining[animal] = max(0, cards_remaining[animal])
    return cards_remaining


def load_all_games(log_dir: str) -> List[List[Dict]]:
    """Load every game-log JSON from log_dir, sorted by filename."""
    games = []
    if not os.path.exists(log_dir):
        return games
    for filename in sorted(os.listdir(log_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    games.append(json.load(f))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return games


class GameState:
    def __init__(self, config: Dict):
        self.config = config
        self.deck = self._create_deck()
        self.player_hand: List[str] = []
        self.ai_hand: List[str] = []
        self.player_face_up: List[str] = []
        self.ai_face_up: List[str] = []
        self.claiming_player = 'player'
        self.current_claim: Optional[str] = None
        self.current_card: Optional[str] = None
        self.phase = 'claim'
        self.game_over = False
        self.winner: Optional[str] = None

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
        for animal in self.config['animals']:
            if self.player_face_up.count(animal) >= threshold:
                return 'ai'
            if self.ai_face_up.count(animal) >= threshold:
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
            'winner': self.winner,
        }


class DataLogger:
    def __init__(self, log_dir: str = 'game_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_game_log: List[Dict] = []
        self.game_count = len([f for f in os.listdir(log_dir) if f.endswith('.json')])

    def log_state_action(self, state: Dict, action: Dict, reward: float = 0):
        self.current_game_log.append({
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': action,
            'reward': reward,
            'turn_number': len(self.current_game_log),
        })

    def save_game(self):
        if self.current_game_log:
            filename = f"game_{self.game_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(os.path.join(self.log_dir, filename), 'w') as f:
                json.dump(self.current_game_log, f, indent=2)
            self.current_game_log = []
            self.game_count += 1
