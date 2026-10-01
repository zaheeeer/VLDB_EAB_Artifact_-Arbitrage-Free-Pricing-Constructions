# Ablation on the main market

Each correction switched on alone from the paper protocol (add_*) and switched off alone from the corrected protocol (drop_*). Seed sets repeat the end points with other sampling seeds; buyer seeds stay the same, so the seed range measures sampling noise only.

End points against the full run (timings excluded):

- paper: identical
- corrected: identical

Switches:

- `C04_demand` (C04): test buyers share the training demand instead of a redrawn popularity ranking
- `C05_fit` (C05): exact, uncapped scale fit instead of a grid capped at 60
- `C06_entropy` (C06): Qirana entropy over real answer classes
- `C07_selection` (C07): selection rewrite from the target's WHERE clause instead of its answer values
- `C09_querymarket` (C09): QueryMarket prices by the cheapest catalogue cover
- `C11_answer_cells` (C11): answer cells counted as rows x columns
- `C12_row_sampling` (C12): any physical row can be deleted, duplicates included
- `alias_resolve` (alias): a deletion is checked against queries that read the table through an alias view
- `remove_all_falsified` (edges): edges refuted by the pricing support are also removed
- `metered_x3` (timing): compute-metered prices from the median of three timings
- `witness_seed` (seed): witness sample drawn independently of the curve sample
- `ilp_formulation` (solver): cover ILP over the cheaper sources only (same optimum; a check)

Not ablated here, because they act only on the other markets (Table 7, Figure 2b): support family of the market runs (C11), empty tables in market samplers, the Table 7 SD (C14), and the full protocol on every market (C15).

## From the paper protocol

| quantity | paper | add_C04_demand | add_C05_fit | add_C06_entropy | add_C07_selection | add_C09_querymarket | add_C11_answer_cells | add_C12_row_sampling | add_alias_resolve | add_remove_all_falsified | add_metered_x3 | add_witness_seed | add_ilp_formulation | paper_seedset1 | paper_seedset2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claimed edges | 2,047 | 2,047 | 2,047 | 2,047 | 1,902 | 2,047 | 2,047 | 2,047 | 2,047 | 2,047 | 2,047 | 2,047 | 2,047 | 2,047 | 2,047 |
| falsified in the curve sample | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 4 | 33 | 33 | 33 | 33 | 11 | 37 |
| witnessed | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 4 | 20 | 20 | 15 | 20 | 8 | 25 |
| falsified by the pricing support | 12 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 31 | 2 |
| false edges total | 45 | 45 | 45 | 45 | 43 | 45 | 45 | 45 | 16 | 45 | 45 | 45 | 45 | 42 | 39 |
| zero-priced (readership) | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 40 | 60 | 60 | 60 | 60 | 53 | 54 |
| Table 6 priceable (size) | 111 | 111 | 111 | 111 | 111 | 111 | 111 | 111 | 133 | 111 | 111 | 111 | 111 | 67 | 122 |
| Table 6 priceable (readership) | 162 | 162 | 162 | 162 | 162 | 162 | 162 | 162 | 182 | 162 | 162 | 162 | 162 | 169 | 168 |
| plans failed | 12 | 17 | 12 | 12 | 12 | 12 | 11 | 12 | 5 | 12 | 12 | 12 | 12 | 4 | 14 |
| fits at the grid cap | 0.237 | 0.250 | 0 | 0.223 | 0.237 | 0.290 | 0.240 | 0.237 | 0.230 | 0.237 | 0.247 | 0.237 | 0.237 | 0.227 | 0.223 |
| Table 4 gaps below zero | 7 | 0 | 3 | 7 | 7 | 8 | 8 | 7 | 5 | 7 | 7 | 7 | 7 | 7 | 7 |
| Table 4 mean gap | 0.034 | 0.129 | 0.105 | 0.034 | 0.034 | 0.016 | 0.043 | 0.034 | 0.068 | 0.034 | 0.033 | 0.034 | 0.034 | 0.039 | 0.035 |
| QueryMarket revenue | 0.088 | 0.090 | 0.529 | 0.088 | 0.087 | 0.062 | 0.085 | 0.088 | 0.092 | 0.087 | 0.088 | 0.088 | 0.088 | 0.092 | 0.088 |
| QueryMarket welfare | 1.706 | 1.574 | 1.069 | 1.706 | 1.707 | 1.554 | 1.768 | 1.706 | 1.702 | 1.707 | 1.706 | 1.706 | 1.706 | 1.702 | 1.706 |
| Qirana max gain | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.780 | 1 | 1 | 1 | 1 | 1 | 1 |

## From the corrected protocol

| quantity | corrected | drop_C04_demand | drop_C05_fit | drop_C06_entropy | drop_C07_selection | drop_C09_querymarket | drop_C11_answer_cells | drop_C12_row_sampling | drop_alias_resolve | drop_remove_all_falsified | drop_metered_x3 | drop_witness_seed | drop_ilp_formulation | corrected_seedset1 | corrected_seedset2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claimed edges | 1,902 | 1,902 | 1,902 | 1,902 | 2,047 | 1,902 | 1,902 | 1,902 | 1,902 | 1,902 | 1,902 | 1,902 | 1,902 | 1,902 | 1,902 |
| falsified in the curve sample | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 33 | 4 | 4 | 4 | 4 | 14 | 13 |
| witnessed | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 15 | 4 | 4 | 4 | 4 | 14 | 13 |
| falsified by the pricing support | 10 | 10 | 10 | 10 | 12 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 3 |
| false edges total | 14 | 14 | 14 | 14 | 16 | 14 | 14 | 14 | 43 | 14 | 14 | 14 | 14 | 14 | 16 |
| zero-priced (readership) | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 60 | 40 | 40 | 40 | 40 | 45 | 35 |
| Table 6 priceable (size) | 133 | 133 | 133 | 133 | 133 | 133 | 133 | 133 | 111 | 133 | 133 | 133 | 133 | 119 | 126 |
| Table 6 priceable (readership) | 182 | 182 | 182 | 182 | 182 | 182 | 182 | 182 | 162 | 182 | 182 | 182 | 182 | 177 | 187 |
| plans failed | 5 | 6 | 5 | 5 | 5 | 5 | 6 | 5 | 17 | 5 | 5 | 5 | 5 | 2 | 16 |
| fits at the grid cap | 0 | 0 | 0.230 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Table 4 gaps below zero | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Table 4 mean gap | 0.143 | 0.116 | 0.106 | 0.144 | 0.143 | 0.143 | 0.147 | 0.143 | 0.147 | 0.143 | 0.142 | 0.143 | 0.143 | 0.136 | 0.142 |
| QueryMarket revenue | 0.472 | 0.534 | 0.066 | 0.472 | 0.472 | 0.472 | 0.453 | 0.472 | 0.479 | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 | 0.472 |
| QueryMarket welfare | 0.999 | 1.125 | 1.445 | 0.999 | 0.999 | 0.999 | 0.963 | 0.999 | 1.018 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| Qirana max gain | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 1 | 0.780 | 0.780 | 0.780 | 0.780 | 0.743 | 0.733 |

ablation_long.csv has every quantity (Tables 3 to 5 per mechanism included), the change against the base, the seed range of the base, and whether the change is larger than that range (sampled quantities only). With few seed sets the range is a rough guide, not a test.
