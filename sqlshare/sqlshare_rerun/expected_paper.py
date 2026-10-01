"""SQLShare numbers printed in the paper (the setting of Table 2), for side-by-side
comparison with a new run.

The dictionaries are named after this package's table numbering: TABLE3 to TABLE7 hold
the SQLShare columns of Tables 4 to 8 of the paper. Nothing here is used to compute a
result; the report only prints these values beside a new run.
"""

MECH_ORDER = [
    "uniform", "size_proportional", "compute_metered", "monopoly_per_asset",
    "monotone_constrained", "qirana_weighted_coverage", "qirana_uniform_gain",
    "qirana_shannon", "querymarket_viewcover", "chawla_ubp", "chawla_uip", "chawla_lpip",
    "chawla_cip", "chawla_layering", "chawla_xos",
]

PAPER_NAME = {
    "uniform": "flat fee (= UBP)", "size_proportional": "size proportional",
    "compute_metered": "compute metered", "monopoly_per_asset": "per-asset monopoly",
    "monotone_constrained": "monotone fitted",
    "qirana_weighted_coverage": "Qirana weighted coverage",
    "qirana_uniform_gain": "Qirana uniform gain", "qirana_shannon": "Qirana entropy",
    "querymarket_viewcover": "QueryMarket view cover (dagger)",
    "chawla_ubp": "uniform bundle (UBP)", "chawla_uip": "uniform item (UIP)",
    "chawla_lpip": "LP item (LPIP)", "chawla_cip": "capacity item (CIP)",
    "chawla_layering": "layering", "chawla_xos": "XOS",
}

# Table 4 of the paper: revenue, welfare, served (recomposing buyers), SQLShare columns
TABLE3 = {
    "uniform": (0.600, 1.396, 0.37),
    "size_proportional": (0.372, 1.653, 0.94),
    "compute_metered": (0.279, 1.698, 0.84),
    "monopoly_per_asset": (0.537, 1.452, 0.63),
    "monotone_constrained": (0.434, 1.093, 0.26),
    "qirana_weighted_coverage": (0.366, 1.514, 0.62),
    "qirana_uniform_gain": (0.366, 1.514, 0.62),
    "qirana_shannon": (0.368, 1.512, 0.62),
    "querymarket_viewcover": (0.088, 1.706, 0.85),
    "chawla_ubp": (0.609, 1.289, 0.32),
    "chawla_uip": (0.413, 1.190, 0.49),
    "chawla_lpip": (0.490, 1.463, 0.76),
    "chawla_cip": (0.481, 1.365, 0.65),
    "chawla_layering": (0.181, 1.204, 0.73),
    "chawla_xos": (0.528, 1.286, 0.57),
}

# Table 5 of the paper: generalization gap, SQLShare column
TABLE4 = {
    "uniform": -0.108,
    "size_proportional": -0.139,
    "compute_metered": -0.313,
    "monopoly_per_asset": 0.470,
    "monotone_constrained": 0.411,
    "qirana_weighted_coverage": -0.060,
    "qirana_uniform_gain": -0.060,
    "qirana_shannon": -0.056,
    "querymarket_viewcover": 0.091,
    "chawla_ubp": -0.051,
    "chawla_uip": 0.121,
    "chawla_lpip": 0.058,
    "chawla_cip": 0.013,
    "chawla_layering": 0.108,
    "chawla_xos": 0.019,
}

# Table 6 of the paper: information arbitrage, combination arbitrage, max gain, SQLShare columns
TABLE5 = {
    "uniform": (0.000, 0.000, 0.000),
    "size_proportional": (0.000, 0.000, 0.000),
    "compute_metered": (0.452, 0.347, 0.985),
    "monopoly_per_asset": (0.337, 0.349, 0.958),
    "monotone_constrained": (0.000, 0.000, 0.069),
    "qirana_weighted_coverage": (0.006, 0.005, 1.000),
    "qirana_uniform_gain": (0.006, 0.005, 1.000),
    "qirana_shannon": (0.006, 0.005, 1.000),
    "querymarket_viewcover": (0.001, 0.000, 0.000),
    "chawla_ubp": (0.000, 0.000, 0.000),
    "chawla_uip": (0.006, 0.005, 1.000),
    "chawla_lpip": (0.002, 0.002, 0.500),
    "chawla_cip": (0.008, 0.004, 0.900),
    "chawla_layering": (0.003, 0.002, 0.500),
    "chawla_xos": (0.005, 0.004, 0.900),
}

# Table 7 of the paper: priceable assets, median |C|, price levels, build seconds
TABLE6 = {
    "size_proportional": (111, 0.5, 5, 5.5),
    "readership_weighted": (162, 75, 38, 622),
}

# Table 8 of the paper: revenue mean, sd across markets, welfare mean, sd across markets
TABLE7 = {
    "uniform": (0.700, 0.225, 1.597, 0.594),
    "monopoly_per_asset": (0.728, 0.297, 1.404, 0.504),
    "qirana_weighted_coverage": (0.249, 0.293, 1.940, 0.748),
    "querymarket_viewcover": (0.579, 0.295, 1.869, 0.649),
    "chawla_ubp": (0.714, 0.245, 1.476, 0.526),
    "chawla_lpip": (0.349, 0.352, 1.843, 0.767),
}
