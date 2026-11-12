"""Run the frozen retrospective Company 2 monthly temporal sensitivity."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from linn.experiments.company2_temporal import run

if __name__ == '__main__':
    run(ROOT)
