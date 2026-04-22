from agents.base import Agent
from agents.pomdp_agent import POMDPAgent

try:
    from agents.dqn_agent import DQNAgent
except ImportError:
    pass  # torch not installed
