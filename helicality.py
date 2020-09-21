import json
import os

MAIN_DIR = "/Users/sripathisridhar/Documents/GitHub/embedding-bio"

def helicality(file_path):
    '''
    Compute helicality from a JSON file containing a dict of Euclidean losses, store in new JSON file

    Inputs
    -----
    file_path : Path to JSON file containing Euclidean losses

    Returns
    -----
    None
    '''

    losses_dict = {}
    with open(file_path, "r") as f:
        losses_dict = json.load(f)
    
    helicality_dict = {k : 1.0 / loss for k, loss in losses_dict.items()}
    file_name = os.path.splitext(file_path)[0]
    dataset_name = file_name.split("_")[0]

    with open("{}_helicality.json".format(dataset_name), "w") as f:
        json.dump(helicality_dict, f)
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset [TinySOL, NTVow, ENST]")
    args = parser.parse_args()

    if args.dataset.lower() not in ["tinysol", "ntvow", "enst"]:
        raise ValueError("invalid argument")

    elif args.dataset.lower() == "tinysol":
        helicality(os.path.join(MAIN_DIR, "TinySOL_euclideanLosses.json"))
    
    elif args.dataset.lower() == "ntvow":
        helicality(os.path.join(MAIN_DIR, "NTVow_euclideanLosses.json"))

    else:
        helicality(os.path.join(MAIN_DIR, "ENST-drums-public_euclideanLosses.json"))

