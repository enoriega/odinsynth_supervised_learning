""" Split the data into train, validation and test """
import json
import random
from collections import defaultdict
import itertools as it
from pathlib import Path

from tqdm import tqdm

def split(path:str, files_dir, num_eval=1_000, num_test=6_000):
    """ Generates the list of ids per split """

    valid_ids = {int(p.stem) for p in tqdm(Path(files_dir).iterdir(), "Inspecting files") if p.suffix == ".tsv"}


    rules = defaultdict(list)
    total_instances = 0
    with open(path) as f:
        for line in tqdm(f, 'Reading json'):
            data = json.loads(line)
            id = data['id']
            if id in valid_ids:
                rule = data['question']
                total_instances += 1
                rules[rule].append(id)

    print(f"Total rules: {len(rules)}")
    # Shuffle the keys
    keys = list(rules.keys())
    random.shuffle(keys)

    # Start writing down the rules to each file for split
    with open("dev.txt", 'w') as f:
        f.writelines(f"{i}\n" for i in it.chain.from_iterable(rules[k] for k in tqdm(keys[:num_eval], desc="Writing dev")))

    with open("test.txt", 'w') as f:
        f.writelines(f"{i}\n" for i in it.chain.from_iterable(rules[k] for k in tqdm(keys[num_eval:num_eval+num_test], desc="Writing test")))


if __name__ == "__main__":
    split("data/merged_train_split_train.jsonl", "train")