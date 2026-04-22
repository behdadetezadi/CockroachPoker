"""Entry point. Run:  python main.py"""

from core import GAME_CONFIG


def _rules():
    print("\n  Rules:")
    print("  - Select a card, then choose what animal to claim")
    print("  - When AI claims, decide: TRUTH or BLUFF")
    print("  - First to get 3 of the same animal face-up LOSES")


def main():
    print("\n" + "=" * 60)
    print("COCKROACH POKER — RL EDITION")
    print("=" * 60)
    print("\n1. Play vs RANDOM AI     (collect training data)")
    print("2. Play vs POMDP AI      (tabular Q-learning, trains from logs)")
    print("3. Play vs SB3 DQN AI    (neural network, trains via self-play)")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        print("\nRandom AI — play 20-30 games, then train POMDP: python train.py pomdp")
        _rules()
        input("\nPress ENTER to start...")
        from ui import CockroachPokerUI
        CockroachPokerUI().run()

    elif choice == '2':
        from agents.pomdp_agent import POMDPAgent
        print("\nLoading POMDP agent...")
        agent = POMDPAgent(GAME_CONFIG)
        s = agent.get_statistics()
        print(f"  States learned: {s['states']}")
        if s['states'] == 0:
            print("  WARNING: not trained yet — will respond randomly.")
            print("  Collect data (option 1) then run: python train.py pomdp")
        else:
            print("  Ready.")
        _rules()
        input("\nPress ENTER to start...")
        from ui import CockroachPokerUI
        CockroachPokerUI(ai_agent=agent).run()
        print("\nRetrain: python train.py pomdp")

    elif choice == '3':
        try:
            from agents.sb3_dqn_agent import SB3DQNAgent
        except ImportError:
            print("stable-baselines3 not installed.")
            print("Install with:  pip install stable-baselines3")
            return
        print("\nLoading SB3 DQN agent...")
        agent = SB3DQNAgent(GAME_CONFIG)
        s = agent.get_statistics()
        if s['trained']:
            print(f"  Timesteps trained: {s['timesteps']:,}")
            print(f"  Ready.")
        else:
            print("  WARNING: not trained yet — will respond randomly.")
            print("  Train first: python train.py sb3 100000")
        _rules()
        input("\nPress ENTER to start...")
        from ui import CockroachPokerUI
        CockroachPokerUI(ai_agent=agent).run()
        print("\nRetrain: python train.py sb3 100000")

    elif choice == '4':
        print("Bye!")

    else:
        print("Invalid choice.")


if __name__ == '__main__':
    main()
