# Supervised Training Data Generator

## `main.py`
This script reads a jsonl file in SQuAD format and computers all the transitions in all the paths from the root node to the _full_ rule.

If a rule has more than five constraints, it is ignored because it becomes to onerous to enumerate all rules.

## `dataset_split.py`
Uses the files generated by `main.py` and the jsonl file to split the dataset in train/dev/test. The output of the script are two files: 

- `dev.txt`
- `test.txt`

Each contains the id of each json object that belongs to the corresponding split. The remaining ids are part of the train split by default.

There is no overlap of rules in the split, but there can be in the specs.