from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin, DataCollatorWithPadding

import itertools as it

from modeling_odinsynth.BertForRuleGeneration import BertForRuleScoringConfig, BertForRuleScoring


@dataclass
class RuleScoringCollator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    include_parent_seqs: bool
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        # Any extra validations after construction will go here
        pass

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # First we need to chain all the feature items
        to_pad = defaultdict(list)
        other_features = defaultdict(list)

        for f in features:
            spec_size = len(f['input_ids'])
            if not self.include_parent_seqs:
                other_features['spec_sizes'].append(spec_size)
            else:
                assert spec_size % 2 == 0, "The number of seqs in the spec is not divisible by 2, bug?"
                half_size = spec_size // 2
                other_features['spec_sizes'].extend([half_size]*2)
            # Split the features
            for k in f:
                if type(f[k]) == list:
                    to_pad[k].extend(f[k])
                elif k == 'labels':
                    # We need to add an "extra" label to account for the parent's spec into
                    # the collated batch. The value is irrelevant, so let's set it to one
                    if self.include_parent_seqs:
                        other_features['labels'].append(1.)
                    # This is the actual label of the (partial) rule to score
                    other_features['labels'].append(f[k])
                else:
                    other_features[k].append(f[k])


        # Make a tensor out of the other features
        other_features = {k:torch.tensor(v) if k != 'spec_sizes' else tuple(v) for k, v in other_features.items()}

        # We are going to put together all the features into a batch and pad to the lenght of the longest sequence
        # Also, return the tensors in the format requested by the user

        padded = self.tokenizer.pad(
            to_pad,
            padding=True,
            pad_to_multiple_of= self.pad_to_multiple_of,
            return_tensors= return_tensors
        )

        return dict(**padded, **other_features)


@dataclass
class RuleSpecEncoder:
    tokenizer: PreTrainedTokenizerBase
    max_seq_length: Optional[int] = None
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

    def __call__(self, examples:Dict[str, Any]) -> Dict[str, Any]:

        # We are going to prepend the rule with each of the sentences in the spec
        ret = defaultdict(list)

        for parent, correct_transition, incorrect_transition, spec, matches in zip(examples['parent'], examples['child'], examples['negative_child'], examples['spec'], examples['matches']):
            seqs = self.__insert_span_tokens(spec, matches)

            positive_batch_encoding = self.__prepend_rule(correct_transition, parent if  self.include_parent else None,  seqs)

            for k, v in positive_batch_encoding.items():
                ret[k].append(v)
            ret['labels'].append(1.)

            negative_batch_encoding =  self.__prepend_rule(incorrect_transition, parent if self.include_parent else None, seqs)

            for k, v in negative_batch_encoding.items():
                ret[k].append(v)
            ret['labels'].append(0.)

        return ret

if __name__ == "__main__":
    ds = load_dataset("enoriega/odinsynth_dataset", split="train")
    ds = ds.select(range(1000))
    ckpt = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.add_special_tokens({"additional_special_tokens":  ["<sp>", "</sp>"]})

    encoder = RuleSpecEncoder(tokenizer=tokenizer,
                              include_parent=True,
                              max_seq_length=tokenizer.model_max_length,
                              max_spec_seqs=4)

    tokenized_ds = ds.map(encoder, batched=True, remove_columns=ds.column_names, num_proc=16)
    print(tokenized_ds)
    # tokenized_ds = tokenized_ds.remove_columns(["labels"])

    collator = RuleScoringCollator(tokenizer=tokenizer, include_parent_seqs=True, pad_to_multiple_of=8)
    # collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=512)
    sample = tokenized_ds.select(range(3))


    config = BertForRuleScoringConfig.from_pretrained(
        ckpt,
        rule_sentence_encoding="max",
        spec_encoding="max",
        loss_func="margin",
        margin=1.
    )

    model = BertForRuleScoring.from_pretrained(ckpt, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(no_cuda=False, output_dir='.'),
        # args=training_args,
        train_dataset=tokenized_ds,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        # compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        # if training_args.do_eval and not is_torch_tpu_available()
        # else None,
    )

    trainer.train()



