import json


def load_dataset(path):
    """
    Load the evaluation dataset from a JSON file.
    """

    with open(path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset