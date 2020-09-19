import json

# TinySOL
with open("TinySOL_euclideanLosses.json", "r") as f:
    tinysol_dict = json.load(f)

tinysol_helicality = {k : 1.0 / v for k, v in tinysol_dict.items()}

with open("TinySOL_helicality.json", "w") as f:
    json.dump(tinysol_helicality, f)

# NTVow
with open("NTVow_euclideanLosses.json", "r") as f:
    ntvow_dict = json.load(f)

ntvow_helicality = {k : 1.0 / v for k, v in ntvow_dict.items()}

with open("NTVow_helicality.json", "w") as f:
    json.dump(ntvow_helicality, f)
