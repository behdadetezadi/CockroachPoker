#!/usr/bin/env python3
"""
Simple setup script for Cockroach Poker AI
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    try:
        # Install basic requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame>=2.1.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python>=4.5.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.21.0"])
        print("✅ Successfully installed all requirements!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def test_installation():
    """Test if everything is working"""
    try:
        import pygame
        import cv2
        import numpy
        print("✅ All modules imported successfully!")

        # Test camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected and working!")
            cap.release()
        else:
            print("⚠️  Camera not detected, will use mock face detection")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    print("🪳 Cockroach Poker AI - Setup")
    print("=" * 40)

    print("Installing requirements...")
    if not install_requirements():
        print("Setup failed!")
        return

    print("\nTesting installation...")
    if not test_installation():
        print("Some components failed, but the game might still work")

    print("\n🎮 Setup complete! Run the game with:")
    print("python main.py")

    print("\n📋 Game Instructions:")
    print("1. The game will try to use your camera for face detection")
    print("2. Click cards to select them")
    print("3. Choose what creature to claim")
    print("4. Try to bluff the AI!")
    print("5. Challenge the AI when it seems to be lying")

    print("\n🤖 The AI learns from your patterns:")
    print("- How long you take to decide")
    print("- Your facial expressions")
    print("- How often you bluff")


if __name__ == "__main__":
    main()