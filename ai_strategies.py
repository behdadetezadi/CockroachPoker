"""
AI Strategies for Cockroach Poker
Multiple different approaches to compare performance
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseAIStrategy(ABC):
    """Base class for all AI strategies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.decision_history = []
        self.player_stats = {
            'total_claims': 0,
            'total_bluffs': 0,
            'avg_decision_time': 2.0,
            'nervous_when_bluffing': 0,
            'nervous_when_truthful': 0,
            'confidence_when_bluffing': 0,
            'confidence_when_truthful': 0,
            'recent_accuracy': 0.5  # How often we've been right lately
        }
        self.games_played = 0
        self.wins = 0

    @abstractmethod
    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        """Decide whether to call BLUFF (True) or TRUE (False)"""
        pass

    @abstractmethod
    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Choose what to claim when passing a card"""
        pass

    def learn_from_outcome(self, was_bluff: bool, opponent_called_bluff: bool, face_data: Dict, decision_time: float):
        """Update learning based on what happened"""
        self.player_stats['total_claims'] += 1
        if was_bluff:
            self.player_stats['total_bluffs'] += 1

        # Update average decision time
        old_avg = self.player_stats['avg_decision_time']
        new_avg = (old_avg * 0.8) + (decision_time * 0.2)
        self.player_stats['avg_decision_time'] = new_avg

        # Learn emotion patterns
        emotion = face_data.get('emotion', 'neutral')
        confidence = face_data.get('confidence', 0.5)

        if emotion == 'nervous':
            if was_bluff:
                self.player_stats['nervous_when_bluffing'] += 1
            else:
                self.player_stats['nervous_when_truthful'] += 1

        # Track confidence patterns
        if was_bluff:
            self.player_stats['confidence_when_bluffing'] += confidence
        else:
            self.player_stats['confidence_when_truthful'] += confidence

        # Determine if opponent was correct
        opponent_correct = (opponent_called_bluff and was_bluff) or (not opponent_called_bluff and not was_bluff)

        # Store decision for analysis
        self.decision_history.append({
            'was_bluff': was_bluff,
            'opponent_called_bluff': opponent_called_bluff,
            'emotion': emotion,
            'confidence': confidence,
            'decision_time': decision_time,
            'opponent_correct': opponent_correct
        })

        # Update recent accuracy
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        if recent_decisions:
            correct_calls = sum(1 for d in recent_decisions if
                                (d['opponent_called_bluff'] and d['was_bluff']) or
                                (not d['opponent_called_bluff'] and not d['was_bluff']))
            self.player_stats['recent_accuracy'] = correct_calls / len(recent_decisions)

        # Keep only recent history
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-30:]

    def get_stats(self) -> Dict:
        """Get strategy performance stats"""
        return {
            'name': self.name,
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.wins / max(1, self.games_played),
            'recent_accuracy': self.player_stats['recent_accuracy'],
            'total_decisions': len(self.decision_history)
        }


class RandomAI(BaseAIStrategy):
    """Random AI - baseline for comparison"""

    def __init__(self):
        super().__init__("Random AI", "Makes completely random decisions")

    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        """50/50 random decision"""
        return random.random() < 0.5

    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Random claim - 50% truth, 50% lie"""
        if random.random() < 0.5:
            return actual_creature
        else:
            creatures = ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
            possible_lies = [c for c in creatures if c != actual_creature]
            return random.choice(possible_lies)


class PatternAI(BaseAIStrategy):
    """Pattern-based AI - learns from timing and emotions"""

    def __init__(self):
        super().__init__("Pattern AI", "Analyzes timing patterns and emotions")

    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        suspicion_score = 0.4  # Start neutral

        # Analyze decision timing
        avg_time = self.player_stats['avg_decision_time']
        if decision_time > avg_time * 1.5:
            suspicion_score += 0.3  # Took too long
        elif decision_time < avg_time * 0.5:
            suspicion_score += 0.2  # Too quick

        # Analyze facial expression
        emotion = face_data.get('emotion', 'neutral')
        confidence = face_data.get('confidence', 0.5)

        if emotion == 'nervous' and confidence > 0.7:
            if self.player_stats['nervous_when_bluffing'] > self.player_stats['nervous_when_truthful']:
                suspicion_score += 0.4
        elif emotion == 'confident' and confidence > 0.8:
            suspicion_score -= 0.15

        # Consider historical bluff rate
        if self.player_stats['total_claims'] > 3:
            bluff_rate = self.player_stats['total_bluffs'] / self.player_stats['total_claims']
            if bluff_rate > 0.6:
                suspicion_score += 0.25
            elif bluff_rate < 0.3:
                suspicion_score -= 0.15

        return random.random() < max(0.1, min(0.9, suspicion_score))

    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Pattern-based claiming"""
        # Tell truth 70% of time
        if random.random() < 0.7:
            return actual_creature

        # When lying, pick random creature
        creatures = ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
        possible_lies = [c for c in creatures if c != actual_creature]
        return random.choice(possible_lies)


class AggressiveAI(BaseAIStrategy):
    """Aggressive AI - tries to hurt the opponent strategically"""

    def __init__(self):
        super().__init__("Aggressive AI", "Targets opponent's weaknesses aggressively")

    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        suspicion_score = 0.6  # Start more suspicious

        # Be more aggressive about calling bluffs
        avg_time = self.player_stats['avg_decision_time']
        if decision_time > avg_time * 1.3:
            suspicion_score += 0.4
        elif decision_time < avg_time * 0.6:
            suspicion_score += 0.3

        # Aggressive emotion analysis
        emotion = face_data.get('emotion', 'neutral')
        confidence = face_data.get('confidence', 0.5)

        if emotion in ['nervous', 'thinking']:
            suspicion_score += 0.3
        elif emotion == 'confident' and confidence < 0.6:
            suspicion_score += 0.4  # Fake confidence

        # Strategic: if human has many of this creature, they're probably lying
        human_counts = game_state.get('human_creature_counts', {})
        creature_count = human_counts.get(claimed_creature, 0)
        if creature_count >= 2:
            suspicion_score += 0.5  # Very suspicious!

        return random.random() < max(0.2, min(0.9, suspicion_score))

    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Aggressive strategic claiming"""
        human_counts = game_state.get('human_creature_counts', {})
        ai_counts = game_state.get('ai_creature_counts', {})

        # Find creatures human has many of
        dangerous_for_human = [creature for creature, count in human_counts.items() if count >= 2]

        # VERY aggressive: 70% chance to target human's weakness
        if dangerous_for_human and random.random() < 0.7:
            target = random.choice(dangerous_for_human)
            print(f"🎯 Aggressive AI targets {target} (human has {human_counts[target]})")
            return target

        # Avoid claiming creatures we have many of
        safe_claims = [c for c in ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
                       if ai_counts.get(c, 0) < 2]

        if safe_claims:
            # 30% truth rate when being strategic
            if random.random() < 0.3 and actual_creature in safe_claims:
                return actual_creature
            else:
                return random.choice(safe_claims)

        # Fallback: tell truth
        return actual_creature


class DefensiveAI(BaseAIStrategy):
    """Defensive AI - plays conservatively"""

    def __init__(self):
        super().__init__("Defensive AI", "Plays conservatively and safely")

    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        suspicion_score = 0.3  # Start less suspicious

        # Conservative timing analysis
        avg_time = self.player_stats['avg_decision_time']
        if decision_time > avg_time * 1.8:  # Only very long times are suspicious
            suspicion_score += 0.25
        elif decision_time < avg_time * 0.3:  # Only very quick times
            suspicion_score += 0.15

        # Conservative emotion analysis
        emotion = face_data.get('emotion', 'neutral')
        confidence = face_data.get('confidence', 0.5)

        if emotion == 'nervous' and confidence > 0.8:  # Only very nervous
            suspicion_score += 0.3

        # Only call bluff if very confident
        return random.random() < max(0.05, min(0.7, suspicion_score))

    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Conservative claiming - mostly truth"""
        ai_counts = game_state.get('ai_creature_counts', {})

        # Almost always tell truth (90%)
        if random.random() < 0.9:
            return actual_creature

        # When lying, avoid dangerous creatures for us
        safe_lies = [c for c in ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
                     if c != actual_creature and ai_counts.get(c, 0) == 0]

        if safe_lies:
            return random.choice(safe_lies)
        else:
            return actual_creature  # Too dangerous to lie


class LearningAI(BaseAIStrategy):
    """Advanced learning AI - adapts based on success rate"""

    def __init__(self):
        super().__init__("Learning AI", "Adapts strategy based on success rate")
        self.strategy_weights = {
            'timing': 1.0,
            'emotion': 1.0,
            'pattern': 1.0,
            'strategic': 1.0
        }

    def should_call_bluff(self, claimed_creature: str, game_state: Dict, face_data: Dict, decision_time: float) -> bool:
        suspicion_score = 0.4

        # Weighted timing analysis
        avg_time = self.player_stats['avg_decision_time']
        timing_suspicion = 0
        if decision_time > avg_time * 1.4:
            timing_suspicion = 0.3
        elif decision_time < avg_time * 0.6:
            timing_suspicion = 0.2
        suspicion_score += timing_suspicion * self.strategy_weights['timing']

        # Weighted emotion analysis
        emotion = face_data.get('emotion', 'neutral')
        confidence = face_data.get('confidence', 0.5)
        emotion_suspicion = 0
        if emotion == 'nervous' and confidence > 0.7:
            emotion_suspicion = 0.4
        elif emotion == 'confident' and confidence < 0.5:
            emotion_suspicion = 0.3
        suspicion_score += emotion_suspicion * self.strategy_weights['emotion']

        # Weighted pattern analysis
        pattern_suspicion = 0
        if self.player_stats['total_claims'] > 0:
            bluff_rate = self.player_stats['total_bluffs'] / self.player_stats['total_claims']
            if bluff_rate > 0.6:
                pattern_suspicion = 0.25
            elif bluff_rate < 0.3:
                pattern_suspicion = -0.15
        suspicion_score += pattern_suspicion * self.strategy_weights['pattern']

        # Weighted strategic analysis
        human_counts = game_state.get('human_creature_counts', {})
        strategic_suspicion = 0
        creature_count = human_counts.get(claimed_creature, 0)
        if creature_count >= 2:
            strategic_suspicion = 0.4
        suspicion_score += strategic_suspicion * self.strategy_weights['strategic']

        return random.random() < max(0.1, min(0.9, suspicion_score))

    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Adaptive claiming strategy"""
        # Adjust truth rate based on recent success
        base_truth_rate = 0.6
        if self.player_stats['recent_accuracy'] > 0.7:
            truth_rate = 0.8  # We're doing well, play safer
        elif self.player_stats['recent_accuracy'] < 0.4:
            truth_rate = 0.4  # We're doing poorly, be more aggressive
        else:
            truth_rate = base_truth_rate

        if random.random() < truth_rate:
            return actual_creature

        # Strategic lying
        human_counts = game_state.get('human_creature_counts', {})
        ai_counts = game_state.get('ai_creature_counts', {})

        # Target human's weaknesses
        dangerous_for_human = [c for c, count in human_counts.items() if count >= 2]
        safe_for_us = [c for c in ['Cockroach', 'Rat', 'Stinkbug', 'Fly', 'Spider', 'Scorpion']
                       if ai_counts.get(c, 0) < 2]

        strategic_targets = [c for c in dangerous_for_human if c in safe_for_us]

        if strategic_targets and random.random() < 0.6:
            return random.choice(strategic_targets)

        # Random safe lie
        safe_lies = [c for c in safe_for_us if c != actual_creature]
        if safe_lies:
            return random.choice(safe_lies)

        return actual_creature

    def learn_from_outcome(self, was_bluff: bool, opponent_called_bluff: bool, face_data: Dict, decision_time: float):
        """Enhanced learning with strategy weight adjustment"""
        super().learn_from_outcome(was_bluff, opponent_called_bluff, face_data, decision_time)

        # Adjust strategy weights based on recent performance
        if len(self.decision_history) >= 5:
            recent = self.decision_history[-5:]

            # Analyze which factors correlated with correct decisions
            timing_success = sum(1 for d in recent if
                                 d['decision_time'] != self.player_stats['avg_decision_time'] and d['opponent_correct'])
            emotion_success = sum(1 for d in recent if d['emotion'] != 'neutral' and d['opponent_correct'])

            # Adjust weights (simplified)
            if timing_success > 3:
                self.strategy_weights['timing'] = min(2.0, self.strategy_weights['timing'] * 1.1)
            elif timing_success < 2:
                self.strategy_weights['timing'] = max(0.5, self.strategy_weights['timing'] * 0.9)

            if emotion_success > 3:
                self.strategy_weights['emotion'] = min(2.0, self.strategy_weights['emotion'] * 1.1)
            elif emotion_success < 2:
                self.strategy_weights['emotion'] = max(0.5, self.strategy_weights['emotion'] * 0.9)


# Available strategies for easy import
AVAILABLE_STRATEGIES = {
    'random': RandomAI,
    'pattern': PatternAI,
    'aggressive': AggressiveAI,
    'defensive': DefensiveAI,
    'learning': LearningAI
}


def get_strategy(strategy_name: str) -> BaseAIStrategy:
    """Get an AI strategy by name"""
    if strategy_name.lower() in AVAILABLE_STRATEGIES:
        return AVAILABLE_STRATEGIES[strategy_name.lower()]()
    else:
        print(f"Unknown strategy '{strategy_name}', defaulting to Random AI")
        return RandomAI()


def get_all_strategies() -> List[BaseAIStrategy]:
    """Get all available strategies for comparison"""
    return [strategy_class() for strategy_class in AVAILABLE_STRATEGIES.values()]