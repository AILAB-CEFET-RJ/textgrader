import json
import os
import random
import sys
from typing import List, Dict, Tuple


def load_json_files(directory: str) -> List[Dict]:
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data.extend(json.load(file))
    return data


def split_data(data: List[Dict], train_percentage: float) -> Tuple[List[Dict], List[Dict]]:
    random.shuffle(data)
    split_index = int(len(data) * train_percentage)
    return data[:split_index], data[split_index:]


def save_json_file(data: List[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main(train_percentage: float, train_file: str, test_file: str):
    if not 0 <= train_percentage <= 1:
        print("Train percentage must be between 0 and 1.")
        sys.exit(1)

    directory = "../../../textgrader-pt-br/jsons"
    data = load_json_files(directory)
    train_data, test_data = split_data(data, train_percentage)

    save_json_file(train_data, train_file)
    save_json_file(test_data, test_file)

    print(f"Data split into {len(train_data)} training and {len(test_data)} testing items.")
    print(f"Training data saved to {os.path.join(directory, train_file)}")
    print(f"Testing data saved to {os.path.join(directory, test_file)}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <train_percentage> <train_file> <test_file>")
        sys.exit(1)

    train_percentage = float(sys.argv[1])
    train_file = sys.argv[2]
    test_file = sys.argv[3]

    main(train_percentage, train_file, test_file)
