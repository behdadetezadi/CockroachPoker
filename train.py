"""
Unified training script for both POMDP and DQN agents.

Usage:
    python train.py              # interactive menu
    python train.py pomdp 10    # train POMDP for 10 epochs
    python train.py dqn 10      # train DQN for 10 epochs
"""

import sys
from core import GAME_CONFIG, load_all_games
from agents.pomdp_agent import POMDPAgent


def train_pomdp(config=None, epochs: int = 10, log_dir: str = 'game_logs'):
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
    print(f"\nDone. States learned: {stats['states']}  |  Total updates: {total}")


def train_dqn(config=None, epochs: int = 10, log_dir: str = 'game_logs'):
    try:
        from agents.dqn_agent import DQNAgent
    except ImportError:
        print("PyTorch not installed — cannot train DQN.")
        return

    config = config or GAME_CONFIG
    agent = DQNAgent(config)

    stats = agent.train_from_logs(log_dir, epochs=epochs)
    if stats['games'] == 0:
        return

    agent.save_model()
    print(f"\nDone. Games: {stats['games']}  |  Experiences: {stats['experiences']}  "
          f"|  Final loss: {stats['final_loss']:.4f}")


def _print_stats(agent_type: str):
    config = GAME_CONFIG
    if agent_type == 'pomdp':
        agent = POMDPAgent(config)
        stats = agent.get_statistics()
        print(f"  States: {stats['states']}")
        print(f"  State-action pairs: {stats['state_action_pairs']}")
        if stats['state_action_pairs']:
            print(f"  Avg Q-value: {stats['avg_q_value']:.4f}")
        print(f"  Belief observations: {stats['belief_state']['total_observations']}")
    else:
        try:
            from agents.dqn_agent import DQNAgent
        except ImportError:
            print("PyTorch not installed.")
            return
        agent = DQNAgent(config)
        stats = agent.get_statistics()
        print(f"  Training steps: {stats['training_step']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Replay buffer: {stats['replay_buffer_size']}")
        print(f"  Parameters: {stats['total_parameters']}")


def main():
    # CLI shortcut: python train.py <agent> <epochs>
    if len(sys.argv) >= 2:
        agent_arg = sys.argv[1].lower()
        epochs = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
        if agent_arg == 'pomdp':
            train_pomdp(epochs=epochs)
        elif agent_arg == 'dqn':
            train_dqn(epochs=epochs)
        else:
            print(f"Unknown agent '{agent_arg}'. Use 'pomdp' or 'dqn'.")
        return

    # Interactive menu
    print("\n" + "=" * 60)
    print("COCKROACH POKER — TRAINING")
    print("=" * 60)
    print("\n1. Train POMDP agent (tabular Q-learning)")
    print("2. Train DQN agent (deep Q-network)")
    print("3. View POMDP statistics")
    print("4. View DQN statistics")
    print("5. Reset POMDP model")
    print("6. Reset DQN model")
    print("7. Exit")

    choice = input("\nEnter choice (1-7): ").strip()

    if choice == '1':
        epochs = input("Training epochs [default 10]: ").strip()
        train_pomdp(epochs=int(epochs) if epochs.isdigit() else 10)

    elif choice == '2':
        epochs = input("Training epochs [default 10]: ").strip()
        train_dqn(epochs=int(epochs) if epochs.isdigit() else 10)

    elif choice == '3':
        _print_stats('pomdp')

    elif choice == '4':
        _print_stats('dqn')

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
        if input("Delete DQN model? (yes/no): ").strip().lower() == 'yes':
            import os
            for f in ('dqn_model.pth', 'dqn_model_metadata.json'):
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
