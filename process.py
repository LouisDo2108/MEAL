from cProfile import label
import json
import os
from pathlib import Path
import shutil


root_data_dir = Path("/home/thuy0050/ft49_scratch/thuy0050/meal/temp")
root_aim_fold = Path("/home/thuy0050/ft49_scratch/thuy0050/meal/aim_folds")

# Specify the path to your JSON file
for fold in range(5):
    for i in ["train", "val"]:
        file_names = []

        file_path = f"/home/thuy0050/ft49_scratch/thuy0050/meal/new_annotations/annotation_4cls_{fold}_{i}.json"

        # Read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)["images"]

        # Extract the "file_name" values
        for item in data:
            file_names.append(Path(item["file_name"]).stem)

        image_folder = root_aim_fold / f"fold_{fold}" / i / "images"
        label_folder = root_aim_fold / f"fold_{fold}" / i / "labels"

        for filename in file_names:
            source_file = (
                root_data_dir / "images" / (Path(filename).stem.split("_")[-1] + ".jpg")
            )
            target_file = image_folder / (filename + ".jpg")
            print("source_file", source_file)
            print("target_file", target_file)
            shutil.copyfile(source_file, target_file)

            source_file = (
                root_data_dir / "labels" / (Path(filename).stem.split("_")[-1] + ".txt")
            )
            target_file = label_folder / (filename + ".txt")
            print("source_file", source_file)
            print("target_file", target_file)
            shutil.copyfile(source_file, target_file)
