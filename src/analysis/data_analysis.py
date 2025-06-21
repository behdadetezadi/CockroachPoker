"""
Data analysis module for bluff detection research
"""

import json
import csv
import statistics
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import time
from datetime import datetime
from pathlib import Path


class BluffAnalyzer:
    """Analyze bluff detection patterns and performance"""

    def __init__(self):
        self.decision_data: List[Dict[str, Any]] = []
        self.session_data: Dict[str, Any] = {}
        self.ai_performance: Dict[str, List[float]] = defaultdict(list)

    def load_data(self, data: List[Dict[str, Any]]):
        """Load decision data for analysis"""
        self.decision_data = data
        self._process_data()

    def _process_data(self):
        """Process loaded data and extract patterns"""
        if not self.decision_data:
            return

        # Group by AI strategy
        self.ai_performance.clear()
        for decision in self.decision_data:
            # Extract AI strategy from context if available
            ai_strategy = decision.get('ai_strategy', 'unknown')

            # Calculate if decision was correct
            was_correct = (
                    (decision['player_challenged'] and decision['was_bluff']) or
                    (not decision['player_challenged'] and not decision['was_bluff'])
            )

            self.ai_performance[ai_strategy].append(1.0 if was_correct else 0.0)

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        if not self.decision_data:
            return {}

        total_decisions = len(self.decision_data)
        correct_decisions = sum(1 for d in self.decision_data
                                if (d['player_challenged'] and d['was_bluff']) or
                                (not d['player_challenged'] and not d['was_bluff']))

        bluffs = [d for d in self.decision_data if d['was_bluff']]
        bluffs_detected = sum(1 for d in bluffs if d['player_challenged'])

        decision_times = [d['decision_time'] for d in self.decision_data if d['decision_time'] > 0]

        return {
            'total_decisions': total_decisions,
            'accuracy': (correct_decisions / total_decisions * 100) if total_decisions > 0 else 0,
            'total_bluffs': len(bluffs),
            'bluffs_detected': bluffs_detected,
            'bluff_detection_rate': (bluffs_detected / len(bluffs) * 100) if bluffs else 0,
            'avg_decision_time': statistics.mean(decision_times) if decision_times else 0,
            'median_decision_time': statistics.median(decision_times) if decision_times else 0,
            'decision_time_std': statistics.stdev(decision_times) if len(decision_times) > 1 else 0
        }

    def analyze_timing_patterns(self) -> Dict[str, Any]:
        """Analyze decision timing patterns"""
        if not self.decision_data:
            return {}

        # Categorize decision times
        fast_decisions = []  # < 1 second
        normal_decisions = []  # 1-3 seconds
        slow_decisions = []  # > 3 seconds

        for decision in self.decision_data:
            time_val = decision['decision_time']
            if time_val < 1.0:
                fast_decisions.append(decision)
            elif time_val > 3.0:
                slow_decisions.append(decision)
            else:
                normal_decisions.append(decision)

        def calculate_accuracy(decisions):
            if not decisions:
                return 0
            correct = sum(1 for d in decisions
                          if (d['player_challenged'] and d['was_bluff']) or
                          (not d['player_challenged'] and not d['was_bluff']))
            return correct / len(decisions) * 100

        return {
            'fast_decisions': {
                'count': len(fast_decisions),
                'accuracy': calculate_accuracy(fast_decisions),
                'bluff_rate': sum(1 for d in fast_decisions if d['was_bluff']) / len(
                    fast_decisions) * 100 if fast_decisions else 0
            },
            'normal_decisions': {
                'count': len(normal_decisions),
                'accuracy': calculate_accuracy(normal_decisions),
                'bluff_rate': sum(1 for d in normal_decisions if d['was_bluff']) / len(
                    normal_decisions) * 100 if normal_decisions else 0
            },
            'slow_decisions': {
                'count': len(slow_decisions),
                'accuracy': calculate_accuracy(slow_decisions),
                'bluff_rate': sum(1 for d in slow_decisions if d['was_bluff']) / len(
                    slow_decisions) * 100 if slow_decisions else 0
            }
        }

    def analyze_emotion_patterns(self) -> Dict[str, Any]:
        """Analyze facial expression patterns"""
        emotion_data = defaultdict(list)
        confidence_data = []

        for decision in self.decision_data:
            facial_expr = decision.get('facial_expression')
            if facial_expr and isinstance(facial_expr, dict):
                emotion = facial_expr.get('emotion', 'unknown')
                confidence = facial_expr.get('confidence', 0)

                emotion_data[emotion].append(decision)
                confidence_data.append({
                    'confidence': confidence,
                    'was_bluff': decision['was_bluff'],
                    'was_correct': (decision['player_challenged'] and decision['was_bluff']) or
                                   (not decision['player_challenged'] and not decision['was_bluff'])
                })

        # Analyze by emotion
        emotion_analysis = {}
        for emotion, decisions in emotion_data.items():
            bluff_rate = sum(1 for d in decisions if d['was_bluff']) / len(decisions) * 100
            accuracy = sum(1 for d in decisions
                           if (d['player_challenged'] and d['was_bluff']) or
                           (not d['player_challenged'] and not d['was_bluff'])) / len(decisions) * 100

            emotion_analysis[emotion] = {
                'count': len(decisions),
                'bluff_rate': bluff_rate,
                'accuracy': accuracy
            }

        # Analyze confidence correlation
        confidence_analysis = {}
        if confidence_data:
            high_conf = [d for d in confidence_data if d['confidence'] > 0.7]
            low_conf = [d for d in confidence_data if d['confidence'] <= 0.7]

            confidence_analysis = {
                'high_confidence': {
                    'count': len(high_conf),
                    'bluff_rate': sum(1 for d in high_conf if d['was_bluff']) / len(
                        high_conf) * 100 if high_conf else 0,
                    'accuracy': sum(1 for d in high_conf if d['was_correct']) / len(high_conf) * 100 if high_conf else 0
                },
                'low_confidence': {
                    'count': len(low_conf),
                    'bluff_rate': sum(1 for d in low_conf if d['was_bluff']) / len(low_conf) * 100 if low_conf else 0,
                    'accuracy': sum(1 for d in low_conf if d['was_correct']) / len(low_conf) * 100 if low_conf else 0
                }
            }

        return {
            'emotion_patterns': emotion_analysis,
            'confidence_analysis': confidence_analysis
        }

    def analyze_ai_strategy_performance(self) -> Dict[str, Any]:
        """Compare different AI strategy performances"""
        strategy_performance = {}

        for strategy, accuracies in self.ai_performance.items():
            if accuracies:
                strategy_performance[strategy] = {
                    'decisions': len(accuracies),
                    'accuracy': statistics.mean(accuracies) * 100,
                    'consistency': (1 - statistics.stdev(accuracies)) * 100 if len(accuracies) > 1 else 100
                }

        return strategy_performance

    def detect_learning_trends(self) -> Dict[str, Any]:
        """Detect if AI strategies are improving over time"""
        if len(self.decision_data) < 10:
            return {'message': 'Insufficient data for trend analysis'}

        # Split data into chunks to analyze improvement
        chunk_size = max(5, len(self.decision_data) // 4)
        chunks = []

        for i in range(0, len(self.decision_data), chunk_size):
            chunk = self.decision_data[i:i + chunk_size]
            accuracy = sum(1 for d in chunk
                           if (d['player_challenged'] and d['was_bluff']) or
                           (not d['player_challenged'] and not d['was_bluff'])) / len(chunk) * 100
            chunks.append(accuracy)

        # Calculate trend
        if len(chunks) >= 2:
            trend = chunks[-1] - chunks[0]  # Simple difference
            improving = trend > 0
        else:
            trend = 0
            improving = False

        return {
            'chunk_accuracies': chunks,
            'trend': trend,
            'improving': improving,
            'latest_accuracy': chunks[-1] if chunks else 0
        }

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        stats = self.get_overall_statistics()
        timing = self.analyze_timing_patterns()
        emotions = self.analyze_emotion_patterns()

        # Accuracy recommendations
        if stats.get('accuracy', 0) < 60:
            recommendations.append(
                "Overall accuracy is low. Consider improving AI strategy or collecting more training data.")

        # Timing recommendations
        if timing:
            fast_accuracy = timing['fast_decisions']['accuracy']
            slow_accuracy = timing['slow_decisions']['accuracy']

            if fast_accuracy > slow_accuracy + 10:
                recommendations.append(
                    "Fast decisions are more accurate. Consider weighting quick decisions more heavily.")
            elif slow_accuracy > fast_accuracy + 10:
                recommendations.append(
                    "Slow decisions are more accurate. Deliberation time may be important for bluff detection.")

        # Emotion recommendations
        emotion_patterns = emotions.get('emotion_patterns', {})
        if 'nervous' in emotion_patterns:
            nervous_bluff_rate = emotion_patterns['nervous']['bluff_rate']
            if nervous_bluff_rate > 70:
                recommendations.append(
                    "'Nervous' emotion strongly correlates with bluffing. Increase weight of nervousness detection.")

        # Bluff detection recommendations
        if stats.get('bluff_detection_rate', 0) < 50:
            recommendations.append("Bluff detection rate is low. Focus on improving false negative reduction.")

        return recommendations


class DataExporter:
    """Export game data for external analysis"""

    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], filename: str):
        """Export decision data to CSV"""
        if not data:
            return

        fieldnames = [
            'timestamp', 'claimed_creature', 'actual_creature', 'was_bluff',
            'player_challenged', 'decision_time', 'game_round', 'confidence_score'
        ]

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for decision in data:
                # Flatten facial expression data
                row = {key: decision.get(key, '') for key in fieldnames}

                # Handle facial expression
                facial_expr = decision.get('facial_expression')
                if facial_expr and isinstance(facial_expr, dict):
                    row['emotion'] = facial_expr.get('emotion', '')
                    row['expression_confidence'] = facial_expr.get('confidence', 0)

                writer.writerow(row)

    @staticmethod
    def export_to_json(data: List[Dict[str, Any]], filename: str):
        """Export decision data to JSON"""
        # Convert any non-serializable objects
        serializable_data = []
        for decision in data:
            serializable_decision = {}
            for key, value in decision.items():
                if key == 'timestamp':
                    serializable_decision[key] = datetime.fromtimestamp(value).isoformat() if isinstance(value, (
                    int, float)) else value
                else:
                    serializable_decision[key] = value
            serializable_data.append(serializable_decision)

        with open(filename, 'w') as jsonfile:
            json.dump(serializable_data, jsonfile, indent=2, default=str)

    @staticmethod
    def export_analysis_report(analyzer: BluffAnalyzer, filename: str):
        """Export comprehensive analysis report"""
        report = {
            'generation_time': datetime.now().isoformat(),
            'overall_statistics': analyzer.get_overall_statistics(),
            'timing_patterns': analyzer.analyze_timing_patterns(),
            'emotion_patterns': analyzer.analyze_emotion_patterns(),
            'ai_strategy_performance': analyzer.analyze_ai_strategy_performance(),
            'learning_trends': analyzer.detect_learning_trends(),
            'recommendations': analyzer.generate_recommendations()
        }

        with open(filename, 'w') as jsonfile:
            json.dump(report, jsonfile, indent=2, default=str)


class MLDataPreprocessor:
    """Prepare data for machine learning models"""

    def __init__(self):
        self.feature_columns = [
            'decision_time', 'game_round', 'confidence_score',
            'emotion_nervous', 'emotion_confident', 'emotion_thinking',
            'emotion_neutral', 'emotion_focused', 'emotion_uncertain',
            'expression_confidence', 'time_category_fast', 'time_category_normal', 'time_category_slow'
        ]

    def prepare_features(self, data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Prepare feature vectors for ML models"""
        features = []

        for decision in data:
            feature_vector = {}

            # Numerical features
            feature_vector['decision_time'] = min(decision.get('decision_time', 0), 10.0)  # Cap at 10 seconds
            feature_vector['game_round'] = decision.get('game_round', 0)
            feature_vector['confidence_score'] = decision.get('confidence_score', 0)

            # Facial expression features
            facial_expr = decision.get('facial_expression', {})
            if isinstance(facial_expr, dict):
                emotion = facial_expr.get('emotion', 'neutral')
                expr_confidence = facial_expr.get('confidence', 0)

                # One-hot encode emotions
                emotions = ['nervous', 'confident', 'thinking', 'neutral', 'focused', 'uncertain']
                for emo in emotions:
                    feature_vector[f'emotion_{emo}'] = 1.0 if emotion == emo else 0.0

                feature_vector['expression_confidence'] = expr_confidence
            else:
                # Default values if no facial expression data
                emotions = ['nervous', 'confident', 'thinking', 'neutral', 'focused', 'uncertain']
                for emo in emotions:
                    feature_vector[f'emotion_{emo}'] = 0.0
                feature_vector['expression_confidence'] = 0.5

            # Time category features
            decision_time = feature_vector['decision_time']
            feature_vector['time_category_fast'] = 1.0 if decision_time < 1.0 else 0.0
            feature_vector['time_category_normal'] = 1.0 if 1.0 <= decision_time <= 3.0 else 0.0
            feature_vector['time_category_slow'] = 1.0 if decision_time > 3.0 else 0.0

            # Target variable
            feature_vector['is_bluff'] = 1.0 if decision.get('was_bluff', False) else 0.0

            features.append(feature_vector)

        return features

    def export_for_sklearn(self, data: List[Dict[str, Any]], filename: str):
        """Export data in format ready for scikit-learn"""
        features = self.prepare_features(data)

        if not features:
            return

        # Create X (features) and y (target) arrays
        X = []
        y = []

        for feature_vector in features:
            x_row = [feature_vector[col] for col in self.feature_columns]
            X.append(x_row)
            y.append(feature_vector['is_bluff'])

        # Save as CSV with headers
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write headers
            headers = self.feature_columns + ['target']
            writer.writerow(headers)

            # Write data
            for i, x_row in enumerate(X):
                row = x_row + [y[i]]
                writer.writerow(row)


# Example usage and testing functions
def example_analysis():
    """Example of how to use the analysis tools"""
    # Mock data for demonstration
    mock_data = [
        {
            'timestamp': time.time(),
            'claimed_creature': 'cockroach',
            'actual_creature': 'rat',
            'was_bluff': True,
            'player_challenged': True,
            'decision_time': 2.5,
            'facial_expression': {'emotion': 'nervous', 'confidence': 0.8},
            'game_round': 1,
            'confidence_score': 0.7
        },
        {
            'timestamp': time.time(),
            'claimed_creature': 'spider',
            'actual_creature': 'spider',
            'was_bluff': False,
            'player_challenged': False,
            'decision_time': 1.2,
            'facial_expression': {'emotion': 'confident', 'confidence': 0.9},
            'game_round': 2,
            'confidence_score': 0.3
        }
    ]

    # Analyze data
    analyzer = BluffAnalyzer()
    analyzer.load_data(mock_data)

    print("Overall Statistics:")
    print(json.dumps(analyzer.get_overall_statistics(), indent=2))

    print("\nTiming Patterns:")
    print(json.dumps(analyzer.analyze_timing_patterns(), indent=2))

    print("\nRecommendations:")
    for rec in analyzer.generate_recommendations():
        print(f"- {rec}")


if __name__ == "__main__":
    example_analysis()