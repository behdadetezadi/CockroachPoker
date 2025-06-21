"""
Biometric detection and analysis

Contains modules for:
- Facial expression detection and emotion analysis
- Decision timing analysis
- Pattern recognition in human behavior
"""

from .face_detection import FaceDetector, EmotionAnalyzer, FaceDetectionConfig

__all__ = [
    'FaceDetector', 'EmotionAnalyzer', 'FaceDetectionConfig'
]