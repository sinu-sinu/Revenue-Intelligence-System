"""
Convenience wrapper for running model training.

Usage:
    python train.py
    python train.py --data-dir /path/to/data
    python train.py --no-save
"""

from models.training.train_pipeline import main

if __name__ == "__main__":
    main()

