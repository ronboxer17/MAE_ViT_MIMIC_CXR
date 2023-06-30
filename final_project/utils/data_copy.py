import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from final_project.config import MIMIC_FILES_PATH
import json
def process_file(item: Dict[str, str]):
    label = item.get("label")
    path = os.path.join(MIMIC_FILES_PATH, item.get("path"))

    if int(label) == 0:
        destination_folder = "0"
    elif int(label) == 1:
        destination_folder = "1"
    else:
        print(f"Invalid label: {label}. Skipping file: {path}")
        return

    DESTENTION_PATH = r'D:\ron\mimic\final_project\final_project\assets\data\mimic-cxr\mimic_sample\val'

    filename = os.path.basename(path)
    destination_path = os.path.join(DESTENTION_PATH, destination_folder, filename)

    shutil.copy2(path, destination_path)
    # print(f"File '{filename}' saved to '{destination_path}'")


def save_files_based_on_label(data_list: List[Dict[str, Any]]):
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_file, data_list)



json_file_path = r'D:\ron\mimic\final_project\final_project\utils\val_sample_2000.json'

with open(json_file_path) as file:
    data = json.load(file)

if isinstance(data, str):
    data = json.loads(data)

save_files_based_on_label(data)
