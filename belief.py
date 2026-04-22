from collections import defaultdict
from typing import Dict, List


class BeliefState:
    """
    Tracks the AI's beliefs about hidden information (POMDP).

    Maintains:
    - A probability distribution over what animals the opponent likely holds
    - A behavioural model of how often the opponent tells truth vs bluffs per animal
    """

    def __init__(self, animals: List[str]):
        self.animals = animals
        self.opponent_hand_belief: Dict[str, float] = {
            animal: 1.0 / len(animals) for animal in animals
        }
        self.claim_history: List[Dict] = []
        self.truth_count: Dict[str, int] = defaultdict(int)
        self.bluff_count: Dict[str, int] = defaultdict(int)

    def update_after_reveal(self, claim: str, actual_card: str, cards_remaining: Dict[str, int]):
        """
        Update belief and behavioural model after a card is revealed.

        Uses a weighted blend (70% new evidence, 30% prior) so that behavioural
        information accumulated earlier in the game is not discarded entirely.
        """
        was_truth = (claim == actual_card)

        if was_truth:
            self.truth_count[claim] += 1
        else:
            self.bluff_count[claim] += 1

        self.claim_history.append({
            'claim': claim,
            'actual': actual_card,
            'was_truth': was_truth,
        })

        if actual_card in self.opponent_hand_belief:
            total_possible = sum(cards_remaining.values())
            if total_possible > 0:
                for animal in self.animals:
                    new_freq = cards_remaining.get(animal, 0) / total_possible
                    self.opponent_hand_belief[animal] = (
                        0.7 * new_freq + 0.3 * self.opponent_hand_belief[animal]
                    )

    def get_truth_probability(self, claim: str) -> float:
        """
        Estimate P(claim is truthful) by combining:
        - Historical truth rate for this animal
        - Current hand-composition belief
        History is weighted more as more data is collected (up to 70%).
        """
        total_claims = self.truth_count[claim] + self.bluff_count[claim]
        if total_claims > 0:
            historical_truth_rate = self.truth_count[claim] / total_claims
        else:
            historical_truth_rate = 0.5

        hand_belief = self.opponent_hand_belief.get(claim, 0.5)
        history_weight = min(total_claims / 10.0, 0.7)
        belief_weight = 1.0 - history_weight

        return historical_truth_rate * history_weight + hand_belief * belief_weight

    def get_recent_bluff_rate(self, window: int = 5) -> float:
        """Bluff rate over the last `window` claims."""
        recent = (
            self.claim_history[-window:]
            if len(self.claim_history) >= window
            else self.claim_history
        )
        if not recent:
            return 0.5
        bluffs = sum(1 for h in recent if not h['was_truth'])
        return bluffs / len(recent)

    def copy(self) -> 'BeliefState':
        """Return a shallow copy suitable for look-ahead / training."""
        new = BeliefState(self.animals)
        new.opponent_hand_belief = self.opponent_hand_belief.copy()
        new.truth_count = defaultdict(int, self.truth_count)
        new.bluff_count = defaultdict(int, self.bluff_count)
        new.claim_history = list(self.claim_history)
        return new

    def to_dict(self) -> Dict:
        return {
            'hand_belief': dict(self.opponent_hand_belief),
            'truth_counts': dict(self.truth_count),
            'bluff_counts': dict(self.bluff_count),
            'total_claims': len(self.claim_history),
        }
