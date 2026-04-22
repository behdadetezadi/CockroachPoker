"""
Unified training script.

Usage (CLI):
    python train.py pomdp 10          # train POMDP for 10 epochs from game logs
    python train.py sb3 100000        # train SB3 DQN for 100 000 env steps

Usage (interactive):
    python train.py
"""

import sys
from core import GAME_CONFIG, load_all_games
from agents.pomdp_agent import POMDPAgent


def train_pomdp(config=None, epochs: int = 10, log_dir: str = 'game_logs'):
    """Offline Q-learning from saved game logs."""
    config = config or GAME_CONFIG
    agent = POMDPAgent(config)

    games = load_all_games(log_dir)
    if not games:
        print("No training data found — play some games first (option 1 in main.py).")
        return

    print(f"\nTraining POMDP agent on {len(games)} games for {epochs} epoch(s)...")
    total = 0
    for epoch in range(epochs):
        updates = agent.train_from_logs(log_dir)
        total += updates
        print(f"  Epoch {epoch + 1}/{epochs}: {updates} Q-updates")

    agent.save_model()
    stats = agent.get_statistics()
    print(f"\nDone.  States learned: {stats['states']}  |  Total Q-updates: {total}")


def train_sb3(config=None, total_timesteps: int = 100_000):
    """
    Online DQN training via stable-baselines3.

    The agent plays against a random opponent in the headless CockroachPokerEnv.
    No game logs needed — SB3 generates its own experience.
    """
    try:
        from agents.sb3_dqn_agent import SB3DQNAgent
    except ImportError:
        print("Could not import SB3DQNAgent.")
        print("Install dependencies:  pip install stable-baselines3 gymnasium numpy")
        return

    config = config or GAME_CONFIG
    agent = SB3DQNAgent(config)

    print(f"\nTraining SB3 DQN for {total_timesteps:,} timesteps via self-play...")
    print("(SB3 handles replay buffer, target network, epsilon decay automatically)\n")
    agent.train(total_timesteps=total_timesteps)

    stats = agent.get_statistics()
    if stats['trained']:
        print(f"\nDone.  Timesteps: {stats['timesteps']:,}  |  Policy: {stats['policy']}")


def _print_pomdp_stats():
    agent = POMDPAgent(GAME_CONFIG)
    s = agent.get_statistics()
    print(f"  States: {s['states']}")
    print(f"  State-action pairs: {s['state_action_pairs']}")
    if s['state_action_pairs']:
        print(f"  Avg Q-value: {s['avg_q_value']:.4f}")
    print(f"  Belief observations: {s['belief_state']['total_observations']}")


def _print_sb3_stats():
    try:
        from agents.sb3_dqn_agent import SB3DQNAgent
    except ImportError:
        print("stable-baselines3 not installed.")
        return
    s = SB3DQNAgent(GAME_CONFIG).get_statistics()
    if s['trained']:
        print(f"  Timesteps trained: {s['timesteps']:,}")
        print(f"  Policy: {s['policy']}")
        print(f"  Device: {s['device']}")
    else:
        print("  Not trained yet.")


def main():
    # CLI shortcut
    if len(sys.argv) >= 2:
        agent_arg = sys.argv[1].lower()
        if agent_arg == 'pomdp':
            epochs = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
            train_pomdp(epochs=epochs)
        elif agent_arg == 'sb3':
            steps = int(sys.argv[2]) if len(sys.argv) >= 3 else 100_000
            train_sb3(total_timesteps=steps)
        else:
            print(f"Unknown agent '{agent_arg}'. Use 'pomdp' or 'sb3'.")
        return

    print("\n" + "=" * 60)
    print("COCKROACH POKER — TRAINING")
    print("=" * 60)
    print("\n1. Train POMDP agent      (offline, from game logs)")
    print("2. Train SB3 DQN agent    (online self-play, no logs needed)")
    print("3. View POMDP statistics")
    print("4. View SB3 DQN statistics")
    print("5. Reset POMDP model")
    print("6. Reset SB3 DQN model")
    print("7. Exit")

    choice = input("\nEnter choice (1-7): ").strip()

    if choice == '1':
        epochs = input("Training epochs [default 10]: ").strip()
        train_pomdp(epochs=int(epochs) if epochs.isdigit() else 10)

    elif choice == '2':
        steps = input("Timesteps [default 100000]: ").strip()
        train_sb3(total_timesteps=int(steps) if steps.isdigit() else 100_000)

    elif choice == '3':
        _print_pomdp_stats()

    elif choice == '4':
        _print_sb3_stats()

    elif choice == '5':
        if input("Delete POMDP model? (yes/no): ").strip().lower() == 'yes':
            import os
            for f in ('rl_model.pkl', 'rl_model_metadata.json'):
                try:
                    os.remove(f)
                    print(f"Deleted {f}")
                except FileNotFoundError:
                    pass

    elif choice == '6':
        if input("Delete SB3 DQN model? (yes/no): ").strip().lower() == 'yes':
            import os
            for f in ('sb3_dqn.zip',):
                try:
                    os.remove(f)
                    print(f"Deleted {f}")
                except FileNotFoundError:
                    pass

    elif choice == '7':
        print("Bye!")

    else:
        print("Invalid choice.")


if __name__ == '__main__':
    main()
