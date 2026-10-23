from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract base class for all Cockroach Poker AI agents.

    Every agent must implement three methods that ui.py and env.py call:
    - choose_response: decide truth/bluff when the opponent claims
    - choose_card_and_claim: pick a card and a claim for the AI's own turn
    - update_belief: incorporate new information after a card is revealed
    """

    @abstractmethod
    def choose_response(self, state: dict) -> str: ...

    @abstractmethod
    def choose_card_and_claim(self, hand: list[str]) -> tuple[str | None, str | None]: ...

    @abstractmethod
    def update_belief(self, claim: str, actual_card: str, cards_remaining: dict[str, int]) -> None: ...
