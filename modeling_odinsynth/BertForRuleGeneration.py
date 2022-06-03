from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn, tensor
from torch.nn import MSELoss, HingeEmbeddingLoss
from transformers import BertPreTrainedModel, BertModel, BertConfig, AutoTokenizer, AutoConfig
from transformers.utils import ModelOutput




def mve_loss(parent_scores:tensor, child_scores:tensor, labels:tensor, margin:float = 1.) -> tensor:
    # Child correct:
    # max(0, margin - child_score + parent_score)
    # Child incorrect:
    # max(0, margin - parent_score + child_score)

    zeros = torch.zeros_like(child_scores)

    correct_margin = margin - child_scores  +  parent_scores
    incorrect_margin = margin - parent_scores + child_scores

    correct_margin = torch.stack([zeros, correct_margin], dim=1)
    incorrect_margin = torch.stack([zeros, incorrect_margin], dim=1)

    correct_loss = torch.max(correct_margin, dim=1)[0]
    incorrect_loss = torch.max(incorrect_margin, dim=1)[0]

    loss = torch.where(labels.bool(), correct_loss, incorrect_loss)

    # Maybe parameterize the loss reduction in the future
    return loss.mean()

@dataclass
class  RuleScoringOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: torch.FloatTensor = None
    spec_sizes: list[int] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertForRuleScoringConfig(BertConfig):

    def __init__(
            self,
            rule_sentence_encoding: str = "cls",
            spec_encoding:str = "avg",
            loss_func:str = "mse",
            spec_dropout:float = 0.1,
            margin:Optional[float] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.rule_sentence_encoding = rule_sentence_encoding.lower()
        self.spec_encoding = spec_encoding.lower()
        self.loss_func = loss_func.lower()
        self.spec_dropout = spec_dropout
        self.margin = margin
        self.problem_type = "regression" # Fixed to regression, because we predict a score for each rule - spec pair

class BertForRuleScoring(BertPreTrainedModel):
    def __init__(self, config: BertForRuleScoringConfig):
        super().__init__(config)

        # This will be Robert's ckpt or another BERT/Transformer model
        self.bert = BertModel(config)

        # These are the head's parameters
        spec_dropout = (
            config.spec_dropout if config.spec_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(spec_dropout)
        # This is the regression layer, to predict the encoded spec's score
        self.regressor = nn.Linear(config.hidden_size, 1)

        # TODO implement the attention pooling parameters and mechanism

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            spec_sizes: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], RuleScoringOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Encode each rule-sentence pair
        # Assume each tensor in the batch dimension is a rule-sent pair
        # Spec elements are identified by their spec_id value at the same index of the batch dimension

        rule_sent_encoding = self.config.rule_sentence_encoding
        # Use the [CLS] token to represent each rule-sent pair in the spec
        if rule_sent_encoding == "cls":
            pair_embeds = outputs[1]
        # Do pooling
        elif rule_sent_encoding in {"avg", "max"}:
            last_hidden_states = outputs[0]
            # Use the attention mask to zero out the paddigns

            if rule_sent_encoding == "avg":
                last_hidden_states[attention_mask == 0, :] = 0.
                pair_embeds = torch.div(last_hidden_states.sum(dim=1), attention_mask.sum(dim=1).unsqueeze(-1))
            elif rule_sent_encoding == "max":
                # Had to do this clone operation to avoid breaking autograd
                x = last_hidden_states.clone()
                x[attention_mask == 0, :] = last_hidden_states.min()
                pair_embeds = torch.max(x, dim=1)[0]
            else:
                raise ValueError(f"{rule_sent_encoding} is not a valid rule_sentence_encoding option")
        else:
            raise ValueError(f"{rule_sent_encoding} is not a valid rule_sentence_encoding option")

        spec_encoding = self.config.spec_encoding

        if spec_sizes is not None:
            assert sum(spec_sizes) == pair_embeds.size()[0], "Spec sizes must add up to the number of inputs"
        else:
            spec_sizes = [pair_embeds.size()[0]]

        splits = pair_embeds.split(spec_sizes, dim=0)

        embedds = list()
        for split in splits:

            if spec_encoding == "avg":
                spec = split.mean(dim=0)
            elif spec_encoding == "max":
                spec = split.max(dim=0)[0]
            elif spec_encoding == "attention":
                raise NotImplemented("Spec attention pooling is not implemented yet")
            else:
                raise ValueError(f"{spec_encoding} is not a valid spec encoding option")

            embedds.append(spec)

        # Stack the list of tensors into a single tensor with specs
        embedds = torch.stack(embedds, dim=0)
        # Add the dropout to the input encoding
        embedds = self.dropout(embedds)

        # Compute the rule scores
        scores = self.regressor(embedds)

        loss = None
        if labels is not None:
            scores = scores.squeeze()
            labels = labels.squeeze()
            if self.config.loss_func == "mse":
                loss_fct = MSELoss()
                loss = loss_fct(scores, labels)
            elif self.config.loss_func == "margin" and self.config.margin:
                # We need to uncollate the parent and child scores from the scores tensors
                # Also, consider only the labels of the children
                parent_scores = scores[range(0, len(scores), 2)]
                child_scores = scores[range(1, len(scores), 2)]
                labels = labels[range(1, len(labels), 2)]
                # Custom margin loss
                loss = mve_loss(parent_scores, child_scores, labels, margin = self.config.margin)
            else:
                raise ValueError(f'Loss function must be either "mse" or "margin" and have a default margin value')


        if not return_dict:
            output = (scores, spec_sizes) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RuleScoringOutput(
            loss=loss,
            scores=scores,
            spec_sizes = spec_sizes,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    ckpt = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    config = BertForRuleScoringConfig.from_pretrained(
        ckpt,
        rule_sentence_encoding="max",
        spec_encoding="max",
        loss_func="margin",
        margin=1.
    )

    model = BertForRuleScoring.from_pretrained(ckpt, config=config)

    inputs = tokenizer([
        # First spec
        'RULE1 [SEP] Hello, <sp> how are </sp> you?',
        'RULE1 [SEP] I am fine!!',
        # Second spec
        'RULE2 [SEP] Hello, how are you?',
        'RULE2 [SEP] I am fine!!',
    ], padding=True, return_tensors='pt')

    labels = torch.tensor([1, -1])


    outputs = model(**inputs, spec_sizes=[2, 3], labels = labels)

    print(outputs.scores)
    print(outputs.loss)