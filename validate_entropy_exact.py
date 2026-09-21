"""
Validate actual answer-equivalence classes vs singleton approximation for Entropy pricing.
From Stage 2 of the Revision Plan.
"""
import sys
import duckdb
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "candidate"))

from protocol import partition_entropy

def test_exact_vs_singleton_entropy():
    # Construct a scenario where singleton approximation overestimates entropy.
    # i.e., removing row 1 yields Answer A. Removing row 2 yields Answer A. 
    # Singleton approx treats them as 2 distinct classes. Exact treats them as 1 class.
    
    # Let's say we have N=4 support rows.
    # Row 1 removed -> Answer 1
    # Row 2 removed -> Answer 1
    # Row 3 removed -> Answer 2
    # Row 4 removed -> Answer (Unchanged base)
    
    # Exact classes: [Answer 1, Answer 1, Answer 2, Base]
    # Exact probabilities: [2/4, 1/4, 1/4]
    exact_entropy = partition_entropy(["Ans1", "Ans1", "Ans2", "Base"])
    
    # Singleton approximation (Qirana code):
    # distinguished count = 3
    # Probs: [1/4 (Base), 1/4, 1/4, 1/4]
    singleton_entropy = partition_entropy(["Base", "D1", "D2", "D3"])
    
    print(f"Exact Entropy: {exact_entropy:.4f} bits")
    print(f"Singleton Approx Entropy: {singleton_entropy:.4f} bits")
    
    if exact_entropy < singleton_entropy:
        print("CONFIRMED: Singleton approximation strictly overstates entropy when aggregates collide.")

if __name__ == "__main__":
    test_exact_vs_singleton_entropy()
