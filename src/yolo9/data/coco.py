import os
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "coco.yaml"), "r") as f:
    coco_names = yaml.load(f, Loader=yaml.FullLoader)["names"]
