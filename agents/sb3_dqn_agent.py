import os
import random
from typing import Dict, List, Optional, Tuple

from agents.base import Agent
from belief import BeliefState
from core import GAME_CONFIG
from env import state_to_obs, CockroachPokerEnv


class SB3DQNAgent(Agent):
    """
    DQN agent backed by stable-baselines3.

    Training is online self-play through CockroachPokerEnv — no game logs needed.
    The POMDP agent still trains from logs; this agent trains by playing against
    a random opponent in the headless env.

    After training, inference works through the standard Agent interface so
    ui.py doesn't need to know anything about SB3.
    """

    ACTIONS = ['truth', 'bluff']

    def __init__(self, config: Dict = None, model_path: str = 'sb3_dqn'):
        self.config = config or GAME_CONFIG
        self.model_path = model_path
        self.belief_state = BeliefState(self.config['animals'])
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Agent interface (called by ui.py during a real game)
    # ------------------------------------------------------------------

    def choose_response(self, state: Dict) -> str:
        if self.model is None:
            return random.choice(self.ACTIONS)
        obs = state_to_obs(state, self.config, self.belief_state)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.ACTIONS[int(action)]

    def choose_card_and_claim(self, hand: List[str]) -> Tuple[Optional[str], Optional[str]]:
        if not hand:
            return None, None
        card = random.choice(hand)
        claim = card if random.random() < 0.5 else random.choice(self.config['animals'])
        return card, claim

    def update_belief(self, claim: str, actual_card: str, cards_remaining: Dict):
        self.belief_state.update_after_reveal(claim, actual_card, cards_remaining)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 100_000, **kwargs):
        """
        Train via self-play in CockroachPokerEnv.

        SB3 handles replay buffer, target network, epsilon decay, and logging.
        Default hyperparameters match the hand-written DQN that was here before.
        """
        try:
            from stable_baselines3 import DQN
        except ImportError:
            print("stable-baselines3 not installed.")
            print("Install with:  pip install stable-baselines3")
            return

        env = CockroachPokerEnv(self.config)

        self.model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            batch_size=64,
            gamma=0.95,
            exploration_fraction=0.3,       # epsilon anneals over first 30% of steps
            exploration_final_eps=0.05,
            target_update_interval=100,
            **kwargs,
        )
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(f"\nModel saved to {self.model_path}.zip")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_model(self):
        path = self.model_path + '.zip'
        if not os.path.exists(path):
            print(f"No SB3 model at {path}. Train first: python train.py sb3")
            return
        try:
            from stable_baselines3 import DQN
            self.model = DQN.load(self.model_path)
            print(f"SB3 DQN loaded ({self.model.num_timesteps:,} timesteps trained)")
        except ImportError:
            print("stable-baselines3 not installed — pip install stable-baselines3")
        except Exception as e:
            print(f"Error loading SB3 model: {e}")

    def get_statistics(self) -> Dict:
        if self.model is None:
            return {'trained': False, 'timesteps': 0}
        return {
            'trained': True,
            'timesteps': self.model.num_timesteps,
            'policy': type(self.model.policy).__name__,
            'device': str(self.model.device),
        }
