from transformers import BertModel
from torch import nn
import torch

class MultiTaskModel(nn.Module):
    def __init__(self, args):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model)
        self.dropout = nn.Dropout(args.dropout)
        self.idiom_embedding = nn.Embedding(args.idiom_vocab_size, args.idiom_hidden_size)
        self.cls_fc = nn.Linear(args.bert_hidden_size*2, 1)
        self.context_fc = nn.Linear(args.bert_hidden_size*2, args.idiom_hidden_size)
        self.register_buffer('enlarged_candidates', torch.arange(args.idiom_vocab_size))

    def vocab(self, mask_states):
        i_embedding = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [mask_states, i_embedding])

    def forward(self, input_ids, token_type_ids, attention_mask, positions, candidate_ids, targets=None, compute_loss=True):
        bert_last_layer_emb = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask).last_hidden_state  # get the last layer's emb, [b,len,bert_h]

        mask_states = bert_last_layer_emb[[i for i in range(len(positions))], positions]
        cls_states = bert_last_layer_emb[:, 0, :]
        context_states = self.context_fc(torch.cat([mask_states, cls_states], dim=-1))
        ex_context_states = torch.unsqueeze(context_states, 1).repeat(1, candidate_ids.shape[1], 1)

        encoded_candidates = self.idiom_embedding(candidate_ids)
        pooled_output = self.dropout(torch.cat([encoded_candidates, ex_context_states], dim=-1))
        logits = self.cls_fc(pooled_output)

        whole_logits = self.vocab(mask_states)
        reshape_logits = logits.view(len(positions), -1)

        if compute_loss and targets!=None:
            # targets represents for the right position in the options
            # while target represent for the right label index in the whole vocab
            criterion = nn.CrossEntropyLoss()
            loss = criterion(reshape_logits, targets)
            target = torch.gather(candidate_ids, dim=1, index=targets.unsqueeze(1))
            whole_loss = criterion(whole_logits, target.squeeze(1))
            return loss, whole_loss
        else:
            cond_logits = torch.gather(whole_logits, dim=1, index=candidate_ids)
            return reshape_logits, whole_logits, cond_logits


