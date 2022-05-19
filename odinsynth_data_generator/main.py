import csv
import heapq
import itertools as it
import json
import random
from concurrent.futures import as_completed, ProcessPoolExecutor

from odinson.ruleutils import *
from odinson.ruleutils import FieldConstraint, AstNode
from odinson.ruleutils import parse_odinson_query
from odinson.ruleutils.oracle import make_minimal_vocabularies
from tqdm import tqdm


def elegible_item(parsed_rule: AstNode):
    constraints = [n for n in parsed_rule.preorder_traversal() if type(n) == FieldConstraint]
    return len(constraints) <= 5


def generate_transitions(datum):
    rule = datum['question']
    parsed_rule = parse_odinson_query(rule)

    #parsed_rule.expand_leftmost_gole(vocab)

    if elegible_item(parsed_rule):

        # Extract vocabulary out of the  rule
        vocab = make_minimal_vocabularies(parsed_rule)

        all_paths = all_paths_from_root(parsed_rule, vocab)
        heap = list()
        seen = set()
        # Yields all the transitions
        for path in all_paths:
            path_length = len(path)
            for transition_ix, (s, e) in enumerate(zip(path, path[1:])):

                # Generate a negative transition
                transitions = [t for t in s.expand_leftmost_hole(vocab) if t != e]
                e_negative = random.choice(transitions) if len(transitions) > 0 else ''

                s = str(s)
                e = str(e)
                e_negative = str(e_negative)

                if (s, e) not in seen:
                    seen.add((s, e))
                    heapq.heappush(heap, (datum['id'], s, e, e_negative, transition_ix + 1, path_length))


        with open(f'train/{datum["id"]}.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(heap)



if __name__ == "__main__":

    ex = ProcessPoolExecutor()
    futures = list()
    with open("data/merged_train_split_train.jsonl") as f:

        for line in tqdm(f, desc="Submitting futures"):
            try:
                datum = json.loads(line)
                if datum['match']:
                    future = ex.submit(generate_transitions, datum)
                    futures.append(future)

            except Exception as e:
                pass

    for _ in tqdm(as_completed(futures), desc="Waiting on futures"):
        pass

    # ex.shutdown(wait=False)









