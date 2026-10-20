"""
Offline Model Training and Serialization Script
Trains DomainClassifier and AdaptiveIntentClassifier and serializes artifacts via joblib.
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.domain_classifier import DomainClassifier
from src.models.intent_classifier import AdaptiveIntentClassifier


def train_and_export_all_models():
    """Retrain and persist all model artifacts to src/artifacts/."""
    print("=" * 60)
    print("Training and serializing DomainClassifier...")
    _ = DomainClassifier(force_retrain=True)
    print("DomainClassifier successfully serialized to src/artifacts/domain_classifier.joblib")

    print("\nTraining and serializing AdaptiveIntentClassifier...")
    _ = AdaptiveIntentClassifier(force_retrain=True)
    print("AdaptiveIntentClassifier successfully serialized to src/artifacts/intent_classifier.joblib")

    print("=" * 60)
    print("All ML artifacts pre-compiled and saved successfully!")


if __name__ == "__main__":
    train_and_export_all_models()
