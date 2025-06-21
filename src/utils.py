"""
Utility functions and helpers
"""

import os
import json
import pickle
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger("cockroach_poker")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if not"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_game_session(data: Dict[str, Any], session_id: str, data_dir: str = "data") -> str:
    """Save game session data"""
    ensure_directory(data_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{session_id}_{timestamp}.json"
    filepath = Path(data_dir) / filename

    # Add metadata
    data['metadata'] = {
        'session_id': session_id,
        'timestamp': timestamp,
        'saved_at': time.time()
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return str(filepath)


def load_game_session(filepath: str) -> Dict[str, Any]:
    """Load game session data"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_ai_model(model: Any, model_name: str, models_dir: str = "models") -> str:
    """Save AI model to file"""
    ensure_directory(models_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = Path(models_dir) / filename

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    return str(filepath)


def load_ai_model(filepath: str) -> Any:
    """Load AI model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_moving_average(values: List[float], window_size: int = 10) -> List[float]:
    """Calculate moving average of values"""
    if len(values) < window_size:
        return values[:]

    averages = []
    for i in range(len(values) - window_size + 1):
        avg = sum(values[i:i + window_size]) / window_size
        averages.append(avg)

    return averages


def normalize_values(values: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """Normalize values to specified range"""
    if not values:
        return []

    current_min = min(values)
    current_max = max(values)

    if current_max == current_min:
        return [min_val] * len(values)

    normalized = []
    for value in values:
        norm_val = (value - current_min) / (current_max - current_min)
        norm_val = min_val + norm_val * (max_val - min_val)
        normalized.append(norm_val)

    return normalized


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for values"""
    import statistics
    import math

    if len(values) < 2:
        return (0.0, 0.0)

    mean = statistics.mean(values)
    std_dev = statistics.stdev(values)
    n = len(values)

    # t-score for 95% confidence (approximation)
    t_score = 1.96 if n > 30 else 2.0

    margin_of_error = t_score * (std_dev / math.sqrt(n))

    return (mean - margin_of_error, mean + margin_of_error)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_card_data(card_data: Dict[str, Any]) -> bool:
    """Validate card data structure"""
    required_fields = ['id', 'creature']
    return all(field in card_data for field in required_fields)


def validate_decision_data(decision_data: Dict[str, Any]) -> bool:
    """Validate decision data structure"""
    required_fields = [
        'timestamp', 'claimed_creature', 'actual_creature',
        'was_bluff', 'player_challenged', 'decision_time'
    ]
    return all(field in decision_data for field in required_fields)


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import pygame

    pygame.init()

    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pygame_version': pygame.version.ver,
        'cpu_count': os.cpu_count(),
        'timestamp': datetime.now().isoformat()
    }


def create_data_directories() -> Dict[str, Path]:
    """Create all necessary data directories"""
    directories = {
        'data': ensure_directory('data'),
        'models': ensure_directory('models'),
        'logs': ensure_directory('logs'),
        'exports': ensure_directory('exports'),
        'sessions': ensure_directory('data/sessions')
    }
    return directories


class PerformanceTimer:
    """Context manager for timing code execution"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.description} took {format_duration(duration)}")

    def get_duration(self) -> Optional[float]:
        """Get duration if timer has been used"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ConfigManager:
    """Manage configuration files"""

    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value

    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(updates)


# Global instances
logger = setup_logging()
config = ConfigManager()

# Example usage
if __name__ == "__main__":
    # Test utility functions
    print("System Info:")
    print(json.dumps(get_system_info(), indent=2))

    # Test timer
    with PerformanceTimer("Test operation"):
        time.sleep(0.1)

    # Test data directories
    dirs = create_data_directories()
    print(f"Created directories: {list(dirs.keys())}")

    # Test configuration
    config.set("test_key", "test_value")
    print(f"Config test: {config.get('test_key')}")