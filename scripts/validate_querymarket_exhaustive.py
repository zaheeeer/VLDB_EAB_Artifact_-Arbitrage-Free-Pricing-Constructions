"""
Validate QueryMarket cover ILP against exhaustive enumeration on small fixtures.
From Stage 2 of the Revision Plan.
"""
import sys
import numpy as np
from pathlib import Path
from itertools import chain, combinations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "candidate"))

from ports import querymarket_prices
from tpc_substrate import min_cost_cover
from probe_tpch_lattice import Asset

def exhaustive_cover_cost(target_idx, assets, view_idx, view_prices):
    """Brute force exact cover to verify ILP."""
    target = assets[target_idx]
    if target_idx in view_idx:
        return view_prices.get(target_idx, float('inf'))
        
    best_cost = float('inf')
    # All subsets of available views
    valid_views = [i for i in view_idx if i != target_idx and assets[i].template == target.template and target.group <= assets[i].group and assets[i].months <= target.months]
    
    for k in range(1, len(valid_views) + 1):
        for subset in combinations(valid_views, k):
            # Check if subset exactly covers target months
            covered_months = []
            valid = True
            for v in subset:
                covered_months.extend(list(assets[v].months))
            if len(covered_months) == len(target.months) and set(covered_months) == set(target.months):
                cost = sum(view_prices[v] for v in subset)
                best_cost = min(best_cost, cost)
                
    return best_cost

def test_qm_exhaustive():
    # Mock assets
    assets = [
        Asset("q1", frozenset(["returnflag"]), frozenset(["jan", "feb", "mar"])), # Target (idx 0)
        Asset("q1", frozenset(["returnflag", "linestatus"]), frozenset(["jan"])), # View 1 (idx 1)
        Asset("q1", frozenset(["returnflag", "linestatus"]), frozenset(["feb"])), # View 2 (idx 2)
        Asset("q1", frozenset(["returnflag", "linestatus"]), frozenset(["mar"])), # View 3 (idx 3)
        Asset("q1", frozenset(["returnflag", "linestatus"]), frozenset(["jan", "feb"])), # View 4 (idx 4)
    ]
    
    view_idx = [1, 2, 3, 4]
    view_prices = {1: 10.0, 2: 15.0, 3: 5.0, 4: 18.0}
    
    # The optimal exact cover for ["jan", "feb", "mar"] is either:
    # {1, 2, 3} -> 10 + 15 + 5 = 30
    # {4, 3} -> 18 + 5 = 23. (Best!)
    
    print("Running exhaustive search...")
    exhaustive_cost = exhaustive_cover_cost(0, assets, view_idx, view_prices)
    
    print("Running ILP QueryMarket port...")
    ilp_prices = querymarket_prices(assets, view_idx, view_prices, min_cost_cover)
    ilp_cost = ilp_prices[0]
    
    print(f"Exhaustive Cost: {exhaustive_cost}")
    print(f"ILP Cost: {ilp_cost}")
    assert abs(exhaustive_cost - ilp_cost) < 1e-9
    print("CONFIRMED: QueryMarket ILP solver matches exhaustive enumeration.")

if __name__ == "__main__":
    test_qm_exhaustive()
