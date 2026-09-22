# Artifact: Query Pricing Under Recomposition

This repository contains the code, experimental configurations, and LaTeX source required to fully reproduce the findings in **"Query Pricing Under Recomposition: An Experimental Evaluation of Arbitrage-Free Pricing Constructions"** (PVLDB EAB Track).

It provides an evaluation substrate that prices 11 constructions from 3 published systems against 4 baselines on both the TPC-H benchmark and a real-world SQLShare query log.

## Hardware and Software Requirements
* **OS:** Linux, macOS, or Windows
* **Compute:** 12 cores, 32 GB RAM recommended for the full campaign.
* **Python:** 3.10+
* **Dependencies:** Defined in `requirements.txt`.

To set up your environment:
```bash
python -m venv vldb_env
source vldb_env/bin/activate  # On Windows: .\vldb_env\Scripts\activate
pip install -r requirements.txt