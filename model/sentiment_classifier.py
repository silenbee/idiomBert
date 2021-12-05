from transformers import BertModel
from torch import nn

class MultiTaskModel(nn.Module):
    def __init__(self, args):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model)
        self.dropout = nn.Dropout(args.dropout)
        self.senti_fc = nn.Linear(args.bert_hidden_size, args.label_num)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        oce_ids is the index of oce data in the batch data, the others are the same
        e.g. oce_ids=[0~6] ocn_ids=[7~17] tnews_ids=[18~31]
        """
        # bert_last_layer_emb, bert_last_layer_pooling_emb = self.bert(
        #     input_ids=input_ids,
        #     token_type_ids=token_type_ids,
        #     attention_mask=attention_mask)  # get the last layer's emb, [b,len,bert_h]
        # bert_last_layer_cls_emb = bert_last_layer_emb[:, 0, :]  # get the cls emb [b,bert_h]

        bert_last_layer_emb = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask).last_hidden_state  # get the last layer's emb, [b,len,bert_h]
        bert_last_layer_cls_emb = bert_last_layer_emb[:, 0, :]  # get the cls emb [b,bert_h]

        # get the prob logits of oce data by the coe data's cls emb [b,bert_h]
        prob_logits = self.senti_fc(bert_last_layer_cls_emb)  # [b, oce_label_num]
        return prob_logits
