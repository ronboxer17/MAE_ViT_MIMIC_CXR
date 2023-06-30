import json

import  json
def print_label_distribution(json_file_path):
    with open(json_file_path) as file:
        data = json.load(file)

    if isinstance(data, str):
        data = json.loads(data)
    label_counts = {}
    total_labels = 0

    for item in data:
        label = item.get("label")
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

        total_labels += 1

    print("Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / total_labels) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")

print_label_distribution(r'D:\ron\mimic\final_project\final_project\utils\train_sample_10000.json')
