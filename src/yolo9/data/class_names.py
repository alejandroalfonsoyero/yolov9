import os
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "coco.yaml"), "r") as f:
    coco_names = yaml.load(f, Loader=yaml.FullLoader)["names"]


hardhat_names = {
    0: "head",
    1: "helmet",
    2: "person"
}


carplate_names = {
    0: "car plate"
}

firesmoke_names = {
    0: "smoke",
    1: "fire"
}

gas_leak_names = {
    0: "gas leak"
}