#!/usr/bin/env python3
"""
AI Strategy Comparison Tool
Runs automated tests between different AI strategies
"""

import random
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    from ai_strategies import get_all_strategies, AVAILABLE_STRATEGIES
except ImportError:
    print("❌ ai_strategies.py not found!")
    exit(1)


@dataclass
class Card:
    creature: str


@dataclass
class Player:
    name: str
    hand: List[Card]
    table_cards: List[Card]

    def has_lost(self) -> bool:
        creature_counts = {}
        for card in self.table_cards:
            creature_counts[card.creature] = creature_counts.get(card.creature, 0) + 1
        return max(creature_counts.values(), default=0) >= 3

    def get_creature_counts(self) -> Dict[str, int]:
        counts = {}
        for card in self.table_cards:
            counts[card.creature] = counts.get(card.creature, 0) + 1
        return counts


class MockFaceDetector:
    """Mock face detection for testing"""

    def get_current_state(self):
        emotions = ['neutral', 'confident', 'nervous', 'thinking', 'focused']
        return {
            'emotion': random.choice(emotions),
            'confidence': random.uniform(0.4, 0.9)
        }


class GameSimulator:
    """Simulates games between AI strategies"""

    def __init__(self):
        self.creatures = ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
        self.face_detector = MockFaceDetector()

    def create_deck(self) -> List[Card]:
        """Create shuffled deck"""
        deck = []
        for creature in self.creatures:
            for _ in range(6):  # 6 of each creature
                deck.append(Card(creature))
        random.shuffle(deck)
        return deck

    def simulate_game(self, ai1_strategy, ai2_strategy, verbose=False) -> str:
        """Simulate a complete game between two AI strategies"""
        # Create deck and deal
        deck = self.create_deck()
        player1 = Player("AI1", deck[:5], [])
        player2 = Player("AI2", deck[5:10], [])

        current_player = player1
        other_player = player2
        current_strategy = ai1_strategy
        other_strategy = ai2_strategy

        turn_count = 0
        max_turns = 50  # Prevent infinite games

        while turn_count < max_turns:
            if not current_player.hand:
                # Current player is out of cards
                if verbose:
                    print(f"{current_player.name} ran out of cards")
                break

            # Current player passes a card
            card = random.choice(current_player.hand)
            current_player.hand.remove(card)

            # Decide what to claim
            game_state = {
                'human_creature_counts': other_player.get_creature_counts(),
                'ai_creature_counts': current_player.get_creature_counts()
            }

            claimed_creature = current_strategy.choose_claim(card.creature, game_state)

            # Other player decides to call bluff or true
            face_data = self.face_detector.get_current_state()
            decision_time = random.uniform(0.5, 3.0)

            # Flip the game state for the other player's perspective
            other_game_state = {
                'human_creature_counts': current_player.get_creature_counts(),
                'ai_creature_counts': other_player.get_creature_counts()
            }

            calls_bluff = other_strategy.should_call_bluff(
                claimed_creature, other_game_state, face_data, decision_time
            )

            is_bluff = card.creature != claimed_creature

            if verbose:
                print(
                    f"Turn {turn_count + 1}: {current_player.name} claims {claimed_creature}, actual: {card.creature}")
                print(f"  {other_player.name} calls: {'BLUFF' if calls_bluff else 'TRUE'}")

            # Resolve the challenge
            if calls_bluff:
                # Called BLUFF
                if is_bluff:
                    # Correct - claimer gets card
                    current_player.table_cards.append(card)
                    if verbose:
                        print(f"  Correct! {current_player.name} gets the card")
                else:
                    # Wrong - challenger gets card
                    other_player.table_cards.append(card)
                    if verbose:
                        print(f"  Wrong! {other_player.name} gets the card")
            else:
                # Called TRUE
                if is_bluff:
                    # Wrong - challenger gets card
                    other_player.table_cards.append(card)
                    if verbose:
                        print(f"  Wrong! {other_player.name} gets the card")
                else:
                    # Correct - claimer gets card
                    current_player.table_cards.append(card)
                    if verbose:
                        print(f"  Correct! {current_player.name} gets the card")

            # Both strategies learn from the outcome
            current_strategy.learn_from_outcome(is_bluff, calls_bluff, face_data, decision_time)
            other_strategy.learn_from_outcome(is_bluff, calls_bluff, face_data, decision_time)

            # Check win conditions
            if current_player.has_lost():
                if verbose:
                    print(f"{current_player.name} loses! (3 of same creature)")
                return other_player.name
            elif other_player.has_lost():
                if verbose:
                    print(f"{other_player.name} loses! (3 of same creature)")
                return current_player.name

            # Switch turns
            current_player, other_player = other_player, current_player
            current_strategy, other_strategy = other_strategy, current_strategy
            turn_count += 1

        # If max turns reached, declare draw or winner by fewer cards on table
        p1_total = len(player1.table_cards)
        p2_total = len(player2.table_cards)

        if p1_total < p2_total:
            return "AI1"
        elif p2_total < p1_total:
            return "AI2"
        else:
            return "Draw"

    def compare_strategies(self, strategy1, strategy2, num_games=100) -> Dict:
        """Compare two strategies over multiple games"""
        print(f"\n🆚 {strategy1.name} vs {strategy2.name}")
        print(f"Playing {num_games} games...")

        results = {"AI1": 0, "AI2": 0, "Draw": 0}
        game_times = []

        start_time = time.time()

        for game_num in range(num_games):
            if game_num % 20 == 0:
                print(f"  Progress: {game_num}/{num_games}")

            game_start = time.time()
            winner = self.simulate_game(strategy1, strategy2)
            game_end = time.time()

            results[winner] += 1
            game_times.append(game_end - game_start)

        total_time = time.time() - start_time

        return {
            'strategy1_name': strategy1.name,
            'strategy2_name': strategy2.name,
            'strategy1_wins': results["AI1"],
            'strategy2_wins': results["AI2"],
            'draws': results["Draw"],
            'total_games': num_games,
            'avg_game_time': sum(game_times) / len(game_times),
            'total_time': total_time,
            'strategy1_win_rate': results["AI1"] / num_games,
            'strategy2_win_rate': results["AI2"] / num_games
        }

    def tournament(self, num_games_per_match=50):
        """Run a tournament between all available strategies"""
        strategies = get_all_strategies()

        if len(strategies) < 2:
            print("❌ Need at least 2 strategies for comparison")
            return

        print(f"\n🏆 AI STRATEGY TOURNAMENT")
        print(f"Strategies: {[s.name for s in strategies]}")
        print(f"Games per match: {num_games_per_match}")
        print("=" * 60)

        results = {}
        for strategy in strategies:
            results[strategy.name] = {'wins': 0, 'losses': 0, 'draws': 0}

        # Round-robin tournament
        total_matches = len(strategies) * (len(strategies) - 1) // 2
        match_count = 0

        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                match_count += 1
                print(f"\nMatch {match_count}/{total_matches}:")

                strategy1 = strategies[i]
                strategy2 = strategies[j]

                match_result = self.compare_strategies(strategy1, strategy2, num_games_per_match)

                # Update tournament results
                results[strategy1.name]['wins'] += match_result['strategy1_wins']
                results[strategy1.name]['losses'] += match_result['strategy2_wins']
                results[strategy1.name]['draws'] += match_result['draws']

                results[strategy2.name]['wins'] += match_result['strategy2_wins']
                results[strategy2.name]['losses'] += match_result['strategy1_wins']
                results[strategy2.name]['draws'] += match_result['draws']

                print(
                    f"  {strategy1.name}: {match_result['strategy1_wins']} wins ({match_result['strategy1_win_rate']:.1%})")
                print(
                    f"  {strategy2.name}: {match_result['strategy2_wins']} wins ({match_result['strategy2_win_rate']:.1%})")
                print(f"  Draws: {match_result['draws']}")
                print(f"  Avg game time: {match_result['avg_game_time']:.2f}s")

        # Print final tournament results
        print("\n" + "=" * 60)
        print("🏆 FINAL TOURNAMENT RESULTS")
        print("=" * 60)

        # Sort by win rate
        sorted_results = sorted(results.items(),
                                key=lambda x: x[1]['wins'] / max(1, x[1]['wins'] + x[1]['losses']),
                                reverse=True)

        for rank, (strategy_name, stats) in enumerate(sorted_results, 1):
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = stats['wins'] / max(1, total_games)

            print(f"{rank}. {strategy_name}")
            print(f"   Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
            print(f"   Win Rate: {win_rate:.1%}")
            print()


def main():
    print("🧠 AI Strategy Comparison Tool")
    print("Available strategies:", list(AVAILABLE_STRATEGIES.keys()))

    simulator = GameSimulator()

    print("\nChoose comparison mode:")
    print("1. Quick comparison (2 strategies, 50 games each)")
    print("2. Full tournament (all strategies)")
    print("3. Single game demo (verbose)")

    try:
        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            strategies = get_all_strategies()
            if len(strategies) < 2:
                print("❌ Need at least 2 strategies")
                return

            print(f"\nAvailable strategies:")
            for i, strategy in enumerate(strategies):
                print(f"{i + 1}. {strategy.name} - {strategy.description}")

            try:
                idx1 = int(input("Select first strategy (number): ")) - 1
                idx2 = int(input("Select second strategy (number): ")) - 1

                if 0 <= idx1 < len(strategies) and 0 <= idx2 < len(strategies) and idx1 != idx2:
                    result = simulator.compare_strategies(strategies[idx1], strategies[idx2], 50)
                    print(f"\n🎯 FINAL RESULT:")
                    print(
                        f"{result['strategy1_name']}: {result['strategy1_wins']} wins ({result['strategy1_win_rate']:.1%})")
                    print(
                        f"{result['strategy2_name']}: {result['strategy2_wins']} wins ({result['strategy2_win_rate']:.1%})")
                    print(f"Draws: {result['draws']}")
                else:
                    print("❌ Invalid strategy selection")
            except ValueError:
                print("❌ Invalid input")

        elif choice == "2":
            simulator.tournament(50)

        elif choice == "3":
            strategies = get_all_strategies()
            if len(strategies) < 2:
                print("❌ Need at least 2 strategies")
                return

            print("Running single demo game...")
            winner = simulator.simulate_game(strategies[0], strategies[1], verbose=True)
            print(f"\n🏆 Winner: {winner}")

        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n⏹️ Comparison stopped")


if __name__ == "__main__":
    main()