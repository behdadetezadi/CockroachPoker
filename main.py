"""
Entry point for the Cockroach Poker game.

Run:  python main.py
"""

from core import GAME_CONFIG


def _show_rules():
    print("\n  Game Rules:")
    print("  - Select a card, then choose what animal to claim")
    print("  - When AI claims, decide: TRUTH or BLUFF")
    print("  - First to get 3 of the same animal face-up LOSES")
    print("  - If hands run out, most face-up cards loses")


def main():
    print("\n" + "=" * 60)
    print("COCKROACH POKER — RL EDITION")
    print("=" * 60)
    print("\n1. Play vs RANDOM AI  (collect training data)")
    print("2. Play vs POMDP AI   (tabular Q-learning)")
    print("3. Play vs DQN AI     (deep Q-network)")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        print("\nRandom AI active — play 20-30 games, then run: python train.py")
        _show_rules()
        input("\nPress ENTER to start...")
        from ui import CockroachPokerUI
        CockroachPokerUI().run()

    elif choice == '2':
        from agents.pomdp_agent import POMDPAgent
        print("\nInitialising POMDP agent...")
        agent = POMDPAgent(GAME_CONFIG)
        stats = agent.get_statistics()
        print(f"  States learned: {stats['states']}")
        print(f"  State-action pairs: {stats['state_action_pairs']}")
        if stats['states'] == 0:
            print("  WARNING: no training data — agent will respond randomly.")
            print("  Play some games first, then run: python train.py")
        else:
            print("  Agent is trained and ready.")
        _show_rules()
        input("\nPress ENTER to start...")
        from ui import CockroachPokerUI
        CockroachPokerUI(ai_agent=agent).run()
        print("\nTo retrain: python train.py")

    elif choice == '3':
        try:
            from agents.dqn_agent import DQNAgent
        except ImportError:
            print("PyTorch not installed — cannot use DQN agent.")
            return
        print("\nInitialising DQN agent...")
        agent = DQNAgent(GAME_CONFIG)
        stats = agent.get_statistics()
        print(f"  Training steps: {stats['training_step']}")
        print(f"  Epsilon: {stats['epsilon']:.3f}")
        print(f"  Replay buffer: {stats['replay_buffer_size']}")
        if stats['training_step'] == 0:
            print("  WARNING: not trained yet — agent will explore randomly.")
            print("  Train first: python train.py dqn")
        else:
            print("  Agent is trained and ready.")
        _show_rules()
        input("\nPress ENTER to start...")
        from ui import CockroachPokerUI
        CockroachPokerUI(ai_agent=agent).run()
        print("\nTo retrain: python train.py dqn")

    elif choice == '4':
        print("Bye!")

    else:
        print("Invalid choice.")


if __name__ == '__main__':
    main()
