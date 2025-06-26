# 🧠 AI Strategy System Guide

The AI strategies have been separated into their own modular system for easy comparison and extension.

## 📁 File Structure

```
main.py                    # Main game (updated to use strategy system)
ai_strategies.py          # All AI strategy implementations
strategy_comparison.py    # Tool to compare strategy performance
AI_STRATEGY_GUIDE.md      # This guide
```

## 🎮 Using Different Strategies in Game

### In-Game Controls:
- **TAB**: Switch to next AI strategy
- **1-5**: Switch directly to strategy 1, 2, 3, 4, or 5
- **R**: Restart game (keeps current strategy)

### Available Strategies:
1. **Random AI** - Makes random decisions (baseline)
2. **Pattern AI** - Learns from timing and emotions
3. **Aggressive AI** - Targets opponent weaknesses aggressively  
4. **Defensive AI** - Plays conservatively and safely
5. **Learning AI** - Adapts strategy weights based on success

## 🔬 Comparing Strategies

### Quick Comparison:
```bash
python strategy_comparison.py
# Choose option 1: Quick comparison
# Select two strategies to test against each other
```

### Full Tournament:
```bash
python strategy_comparison.py
# Choose option 2: Full tournament
# All strategies compete against each other
```

### Single Game Demo:
```bash
python strategy_comparison.py
# Choose option 3: Single game demo
# Watch a detailed game between two AIs
```

## 🏗️ Creating New AI Strategies

### 1. Basic Strategy Structure:

```python
class MyCustomAI(BaseAIStrategy):
    def __init__(self):
        super().__init__("My Custom AI", "Description of what it does")
    
    def should_call_bluff(self, claimed_creature: str, game_state: Dict, 
                         face_data: Dict, decision_time: float) -> bool:
        """Return True for "BLUFF!", False for "TRUE!" """
        # Your logic here
        return random.random() < 0.5
    
    def choose_claim(self, actual_creature: str, game_state: Dict) -> str:
        """Choose what to claim when passing a card"""
        # Your logic here
        return actual_creature  # Always tell truth
```

### 2. Add to Available Strategies:

```python
# In ai_strategies.py, add to AVAILABLE_STRATEGIES dict:
AVAILABLE_STRATEGIES = {
    'random': RandomAI,
    'pattern': PatternAI,
    'aggressive': AggressiveAI,
    'defensive': DefensiveAI,
    'learning': LearningAI,
    'mycustom': MyCustomAI  # Add your strategy here
}
```

## 📊 Strategy Analysis

### Game State Information:
```python
game_state = {
    'human_creature_counts': {'Cockroach': 2, 'Rat': 1, ...},
    'ai_creature_counts': {'Spider': 1, 'Fly': 0, ...}
}
```

### Face Detection Data:
```python
face_data = {
    'emotion': 'nervous',     # neutral, confident, nervous, thinking, focused
    'confidence': 0.85        # How confident the detection is (0-1)
}
```

### Learning from Outcomes:
```python
def learn_from_outcome(self, was_bluff: bool, opponent_called_bluff: bool, 
                      face_data: Dict, decision_time: float):
    # Update your strategy based on what happened
    # was_bluff: True if the claim was a lie
    # opponent_called_bluff: True if opponent called "BLUFF!"
    # Track patterns, adjust weights, etc.
```

## 🎯 Strategy Design Tips

### 1. **Timing Analysis**
```python
# Players who take longer might be deciding to lie
if decision_time > average_time * 1.5:
    suspicion += 0.3
```

### 2. **Emotion Analysis**
```python
# Nervous players might be bluffing
if emotion == 'nervous' and confidence > 0.7:
    suspicion += 0.4
```

### 3. **Strategic Thinking**
```python
# Target creatures opponent has many of
if opponent_creature_count >= 2:
    return that_creature  # Evil strategic lie!
```

### 4. **Pattern Learning**
```python
# Track opponent's historical bluff rate
bluff_rate = total_bluffs / total_claims
if bluff_rate > 0.6:
    suspicion += 0.2  # Known liar
```

## 📈 Performance Metrics

Each strategy tracks:
- **Win Rate**: Games won vs total games
- **Accuracy**: How often they correctly identify bluffs/truths
- **Learning Speed**: How quickly they adapt to patterns
- **Decision Quality**: Strategic value of their choices

### Example Tournament Results:
```
🏆 FINAL TOURNAMENT RESULTS
1. Learning AI
   Wins: 127, Losses: 98, Draws: 25
   Win Rate: 56.4%

2. Aggressive AI  
   Wins: 118, Losses: 105, Draws: 27
   Win Rate: 52.9%

3. Pattern AI
   Wins: 102, Losses: 120, Draws: 28
   Win Rate: 45.9%
```

## 🔧 Advanced Features

### Strategy Weight Adaptation (Learning AI):
```python
# Adjust weights based on success
if timing_analysis_successful:
    timing_weight *= 1.1
else:
    timing_weight *= 0.9
```

### Multi-Factor Decision Making:
```python
# Combine multiple analysis methods
final_suspicion = (
    timing_suspicion * timing_weight +
    emotion_suspicion * emotion_weight +
    pattern_suspicion * pattern_weight +
    strategic_suspicion * strategic_weight
)
```

### Opponent Modeling:
```python
# Build a model of opponent behavior
opponent_profile = {
    'nervous_when_bluffing': True,
    'quick_decisions_when_lying': False,
    'targets_my_weaknesses': True
}
```

## 🚀 Next Steps

1. **Run comparisons** to see which strategies work best
2. **Create your own strategy** with unique approaches
3. **Analyze the results** to understand why strategies succeed/fail
4. **Combine techniques** from successful strategies
5. **Test against human players** to see real-world performance

The modular system makes it easy to experiment with different approaches to bluff detection and strategic play!

---

**Happy AI Strategy Development! 🤖**