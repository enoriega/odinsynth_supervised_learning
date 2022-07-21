import collections
from pprint import pprint
from typing import Dict, Any, Optional
import itertools as it

import torch
from datasets import Dataset
from transformers import Pipeline, pipeline, AutoTokenizer

from modeling_odinsynth.BertForRuleGeneration import BertForRuleScoring, BertForRuleScoringConfig, RuleScoringOutput
from modeling_odinsynth.utils import RuleSpecEncoder


class RuleScoringPipeline(Pipeline):

    def __init__(self,
                 max_spec_seqs:int,
                 max_seq_length: Optional[int] = None,
                 *args, **kwargs):

        # Call the parent constructor to make the tokenizer available
        super().__init__(*args, **kwargs)

        # Instantiate the encoder
        self.rule_spec_encoder = RuleSpecEncoder(
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length if max_seq_length else self.tokenizer.model_max_length,
            max_spec_seqs=max_spec_seqs,
            include_parent=False # This is relevant only during training
            # Add the missing parameters
        )

        # Maybe add the collator

    def _sanitize_parameters(self, **kwargs):
        # Do whatever is necessary for the arguments of the pipeline
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}


    def preprocess(self, inputs, maybe_arg=2):

        inputs = self.rule_spec_encoder.pipeline_call(inputs)

        # Add the spec size to the inputs
        inputs['spec_sizes'] = min(self.rule_spec_encoder.max_spec_seqs, len(inputs['input_ids']))

        # Make torch tensors (when applicable)
        inputs = {k:torch.tensor([v]) if k != 'spec_sizes' else v for k, v in inputs.items()}

        return inputs


    @staticmethod
    def _melt_tensors(model_inputs):
        """ Post-processing of the pre-processing, needs to operate at batch level """
        melted = {}

        # Compute the selection mask to remove padded sequence from dim=0
        selection_mask = []
        counter = 0
        if type(model_inputs['spec_sizes']) is int:
            model_inputs['spec_sizes'] = [model_inputs['spec_sizes']]
        largest_spec = max(model_inputs['spec_sizes'])
        for size in model_inputs['spec_sizes']:
            for ix in range(largest_spec):
                if ix < size:
                    selection_mask.append(counter)
                counter += 1

        # Melt the batch dimension into the first dimension
        for k, v in model_inputs.items():
            if k != 'spec_sizes':
                v = v.view((-1, v.size()[-1]))
                v = v[selection_mask, :]
            melted[k] = v

        return melted

    def _forward(self, model_inputs, **kwargs):
        # Necessary to "melt" the batch dimension into a 2d tensor and to remove the empty padded sequences from the batch
        model_inputs = self._melt_tensors(model_inputs)
        # Run the rule-scorer model
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs, **kwargs):
        # Build a python dictionary or object with the scores
        ret = []
        for k, v in model_outputs.items():
            name = k
            if name == 'scores':
                name = 'score'
            elif name == 'spec_sizes':
                name = 'spec_size'

            ret.append({
                name: v.item()
            })
        return ret


class BaselineRuleScoringPipeline(RuleScoringPipeline):

    def _forward(self, model_inputs, **kwargs):
        """ Here we will do a baseline model that instead of encoding all of the sentences in the stack,
         averages all the scores of each individual score"""
        # Necessary to "melt" the batch dimension into a 2d tensor and to remove the empty padded sequences from the batch
        model_inputs = self._melt_tensors(model_inputs)

        # We will "hack" the scorer to generate individual scores per sentence by manipulating at the spec_sizes tensor
        orig_spec_sizes = model_inputs['spec_sizes']
        unrolled_spec_sizes = tuple(it.chain.from_iterable([1]*size for size in orig_spec_sizes))
        model_inputs['spec_sizes'] = unrolled_spec_sizes

        # Run the rule-scorer model
        outputs = self.model(**model_inputs)

        # Aggregate the outputs and restore the original spec_sizes
        curr = 0
        averaged_scores = list()
        for spec_size in orig_spec_sizes:
            individual_scores = outputs.scores[curr:curr+spec_size]
            avg_score = individual_scores.mean()
            averaged_scores.append(avg_score.unsqueeze(dim=0))
        averaged_scores = torch.concat(averaged_scores, dim=0)

        new_ouputs = RuleScoringOutput(**outputs)
        new_ouputs['scores'] = averaged_scores
        new_ouputs['spec_sizes'] = torch.tensor(orig_spec_sizes)
        return new_ouputs

if __name__ == "__main__":
    # This is a usage example
    checkpoint = "enoriega/rule_learning_margin_1mm_spanpred"
    config = BertForRuleScoringConfig.from_pretrained(checkpoint)
    model = BertForRuleScoring.from_pretrained(checkpoint, config=config)

    # scorer = pipeline(model=checkpoint, pipeline_class=RuleScoringPipeline)
    pipe = BaselineRuleScoringPipeline(model=model,
                               tokenizer=AutoTokenizer.from_pretrained(checkpoint),
                               max_spec_seqs=4, device=-1)

    d = {
        "rule": '[lemma=flower] [word="("] [word=along | word=large]*',
        'spec': [
            'The pollen was collected from a titan arum that just finished flowering at Selby Botanical Gardens in Sarasota , Fla. Caption : Herbarium director Paul Berry speaks to visitors as they crowd around the corpse flower ( Titan Arum ) , a rare and malodorous flowering plant that blooms only two or three times during an average 40-year life span .',
            'Grown primarily for striking , architectural foliage , rather than their insignificant pink or white flowers ( often hidden by the leaves ) , more rex hybrids are available today than ever before , in a staggering assortment of leaf colours , shapes and textures .',
            'The damaged buds fail to open and a failure of flowering ( " failure to color " ) is often the first injury observed .',
            "Often a congregation 's worship space is decorated with special banners and flowers ( such as Easter lilies ) .",
            'All of these cultivars are perfect flowered ( male and female flower parts ) , so a single vine will be fruitful .',
            "Descriptors : phaseolus-vulgaris ; crop-damage ; epilachna-varivestis ; crop-yield ; cropping-systems ; maine Abstract : Field cage studies were conducted during 1989 and 1990 to estimate the yield response of ' Maine Yelloweye ' dry bean , Phaseolus vulgaris L ., to injury by Mexican bean beetle , Epilachna varivestis Mulsant , at the preflowering ( R5 ) and flowering ( R6 ) stages of plant growth in the low-input and conventional cropping systems in Maine .",
            'Sexual feedback , internode elongation and perfect-flowered dwarfs - - Walton C. Galinat At the time the floral primordia are laid down that will eventually reach either tassel or ear , they are perfect flowered ( bisexual ) and below them the vegetative phase is still in the juvenile stage with telescoped internodes and only partly developed leaves .',
            'The time from flowering ( fertilization ) to fruit maturity ranges from 5 to 6 months , depending upon temperatures .',
            'The irrigation treatments were irrigation until flowering ( TAR2 ) , after flowering ( IAR2 ) , full season ( FSI ) , or not at all ( NI ) and the soybean cultivars were Davis and Lloyd .',
            'It should be noted that fretwork decoration , which , in this case , is not confined to the superstructure of the bema doors but covers much of the surface of the doors themselves , spread over the whole of the iconostasis after about the mid-eighteenth century , as Greek Baroque gradually flowered ( see the Introduction ) .',
            'Flower ( 1988 ) adds knowledge of purpose .',
            'But getting back , we again stopped at the border but this time I climbed out to pick some flowers ( stupid ) and oops we had to go to the police station .',
            'If you choose to propagate mature ivy , then find a shoot that isn ’ t flowering ( preferably taking your cuttings outside of the flowering season ) .',
            'Lilac aldehydes were first isolated from the lilac flower oil ( 4 ) and later identified as fragrant components in gardenia flower ( 5 ) , in Platanthera strict ( 6 ) , and in Artemisia pallens ( 7 ) .',
            'fly from one flower ( thread ) and anotrher .',
            'Numerous flower stems rise in summer from the foliage mound to a height of 3 \' bearing wide , airy panicles ( to 20 " long ) of tiny , variably-colored flowers ( tones of gold , silver , purple and green ) which form a cloud over the foliage that is particularly attractive when backlit .',
            'Curse of the Golden Flower ( Dec. 22 ) On the day the crew shot an important scene showing the solemn Chong Yang ritual , which involves the entire Chinese royal family , the production managers repeatedly instructed crew members to turn off their cellphones .',
            'One of the experimentally grown plants that represented a backcross to Dubautia menziesii flowered ( 3-4 above ) .',
            'One of the experimentally grown plants that represented a backcross to Dubautia menziesii flowered ( Fig 5 , note many heads were removed from this plant for chromosome analysis ) .',
            'In the next plenum which the women hold on the Isle of Pines not only will they have oranges and mandarins , but also flowers ( applause ) .'],
        'matches': [[35, 37],
                    [15, 17],
                    [10, 12],
                    [12, 14],
                    [6, 8],
                    [59, 61],
                    [34, 36],
                    [3, 5],
                    [6, 8],
                    [52, 54],
                    [0, 2],
                    [19, 21],
                    [16, 18],
                    [21, 23],
                    [3, 5],
                    [31, 33],
                    [4, 6],
                    [13, 15],
                    [13, 15],
                    [24, 26]],
    }

    d2 = {**d}
    d2['spec'] = d2['spec'][:1]
    d2['matches'] = d2['matches'][:1]

    data = [d2]
    scores = pipe(data, batch_size=1)
    pprint(scores)