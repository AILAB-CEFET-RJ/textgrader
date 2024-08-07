from datetime import datetime
import os
import time
import json


def saving_results(results, label="", cm=None, comp=""):
    today = datetime.now().strftime("%d-%m-%Y-%H-%M")
    results["date"] = today

    folder_path = (
        f"results/{today}-conjunto{results['conjunto']}-{results['epochs']}-epochs-{comp}"
    )
    os.makedirs(folder_path, exist_ok=True)

    if cm is not None:
        cm.to_csv(f"{folder_path}/confusion_matrix.csv", index=False)

    with open(
        f"{folder_path}/results.json",
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(results, arquivo, indent=4)

    print(f"Results saved to {folder_path}/results.json")
