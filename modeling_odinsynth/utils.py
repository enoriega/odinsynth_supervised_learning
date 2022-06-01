from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin

import itertools as it

@dataclass
class RuleSpecCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    max_spec_seqs: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        # Any extra validations after construction will go here
        pass

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # We are going to prepend the rule with each of the sentences in the spec
        return features


@dataclass
class RuleSpecEncoder:
    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int
    pad_to_multiple_of: Optional[int] = None
    max_spec_seqs: Optional[int] = None
    include_parent: bool = False

    start_tag: str = "<sp>"
    end_tag: str = "</sp>"

    def __post_init__(self):
        # Any extra validations after construction will go here
        pass

    def __insert_span_tokens(self, spec:List[str], matches:List[Tuple[int]]):
        seqs = list()
        pairs = it.islice(zip(spec, matches), self.max_spec_seqs) if self.max_spec_seqs else zip(spec, matches)
        for sent, match in pairs:
            tokens = sent.split()
            tokens = tokens[:match[0]] + [self.start_tag] + tokens[match[0]:match[1]] + [self.end_tag] + tokens[match[1]:]
            seqs.append(' '.join(tokens))

        return seqs

    def __prepend_rule(self, rule:str, parent:Optional[str], seqs:list[str]):
        return self.tokenizer.batch_encode_plus(
            list(it.chain(
                zip([parent] * len(seqs), seqs) if parent else [],
                zip([rule] * len(seqs), seqs)
            )),
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            pad_to_multiple_of=self.pad_to_multiple_of)

    def __call__(self, examples:Dict[str, Any], include_parent:bool = False) -> Dict[str, Any]:

        # We are going to prepend the rule with each of the sentences in the spec
        ret = defaultdict(list)

        for parent, correct_transition, incorrect_transition, spec, matches in zip(examples['parent'], examples['child'], examples['negative_child'], examples['spec'], examples['matches']):
            seqs = self.__insert_span_tokens(spec, matches)

            positive_batch_encoding = self.__prepend_rule(correct_transition, parent if  include_parent else None,  seqs)

            for k, v in positive_batch_encoding.items():
                ret[k].append(v)
            ret['labels'].append(1)

            negative_batch_encoding =  self.__prepend_rule(incorrect_transition, parent if include_parent else None, seqs)

            for k, v in negative_batch_encoding.items():
                ret[k].append(v)
            ret['labels'].append(0)

        return ret

if __name__ == "__main__":
    ds = load_dataset("enoriega/odinsynth_dataset", split='train')
    e = ds.select(range(1000))
    "hola"
    ckpt = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.add_special_tokens({"additional_special_tokens":  ["<sp>", "</sp>"]})

    encoder = RuleSpecEncoder(tokenizer=tokenizer, include_parent=False)

    # x = e.map(encoder, batched=True, remove_columns=ds.column_names, num_proc=16)
    # print(x)

