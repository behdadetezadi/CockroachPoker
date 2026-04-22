from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class Agent(ABC):
    """
    Abstract base class for all Cockroach Poker AI agents.

    Every agent must implement three methods that game.py and env.py call:
    - choose_response: decide truth/bluff when the opponent claims
    - choose_card_and_claim: pick a card and a claim for the AI's own turn
    - update_belief: incorporate new information after a card is revealed
    """

    @abstractmethod
    def choose_response(self, state: Dict) -> str:
        """
        Decide whether the current claim is truth or bluff.

        Args:
            state: Observable game state (no current_card — no cheating).
        Returns:
            'truth' or 'bluff'
        """
        ...

    @abstractmethod
    def choose_card_and_claim(self, hand: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Choose a card from hand and declare what animal it is.

        Returns:
            (card, claim) — both None if hand is empty.
        """
        ...

    @abstractmethod
    def update_belief(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]) -> None:
        """
        Update internal belief state after a card is revealed.

        Args:
            claim: What the claimer said the card was.
            actual_card: What the card actually was.
            cards_remaining: Unaccounted cards from the AI's perspective.
        """
        ...
