import torch

def my_collate(batch):
    """
    used in the dataloader, convert the batch data to the general way
    """
    batch = batch[0]
    # judge the type (train or test). the test type will not return the label ids
    type = 'train' if len(batch[0]['data']) == 4 else 'test'
    task_id = batch[0]['task']
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []


    for idx, temp_data in enumerate(batch):
        data = temp_data['data']
        input_ids.append(data[0])
        token_type_ids.append(data[1])
        attention_mask.append(data[2])
        label_ids.append(data[3])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    label_ids = torch.cat(label_ids)

    return (task_id, input_ids, token_type_ids, attention_mask,
                label_ids)

def chid_collate(batch):
    """
    used in the dataloader, convert the batch data to the general way
    """
    batch = batch[0]
    # judge the type (train or test). the test type will not return the label ids
    type = 'train' if len(batch[0]['data']) == 5 else 'test'
    task_id = batch[0]['task']
    input_ids = []
    token_type_ids = []
    attention_mask = []
    candidates = []
    positions = []
    label_ids = []

    for idx, temp_data in enumerate(batch):
        data = temp_data['data']
        input_ids.append(data[0])
        token_type_ids.append(data[1])
        attention_mask.append(data[2])
        candidates.append(data[3])
        positions.append(data[4])
        label_ids.append(data[5])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    # candidates = torch.cat(candidates, dim=0)
    candidates = torch.stack(candidates)
    positions = torch.cat(positions)
    label_ids = torch.cat(label_ids)

    return (task_id, input_ids, token_type_ids, attention_mask,
            candidates, positions, label_ids)


class Args:
    def __init__(self):
        self.init_idiom_emb()

    def init_senti(self):
        self.train_file = r'data/weibo_senti/train.csv'
        self.dev_file =  r'data/weibo_senti/dev.csv'
        self.test_file = r'data/weibo_senti/test.csv'
        self.pretrain_model = r"hfl/chinese-roberta-wwm-ext"
        self.pretrain_model_dir = r'data/pretrain_bert_models/哈工大-roberta-wwm-ext'
        self.save_model_dir = r'data/my_model'
        self.output_dir = r'data/results'
        self.label_num = 2
        self.epoch = 100
        self.batch_size = 128
        self.bert_lr = 3e-5
        self.lr = 3e-4
        self.num_turn = 3
        self.warm_up = 0.1
        self.weight_decay = 0.01
        self.dropout = 0.1
        self.bert_hidden_size = 768
        self.max_length = 160
        self.eval_ratio = 0.1
        self.train_rule = 'Reweight'  #  Reweight or DRW or Resample

    def init_idiom_emb(self):
        self.train_file = r'/home/zeyu/data/chid_standard/train.txt'
        self.dev_file = r'/home/zeyu/data/chid_standard/dev.txt'
        self.test_file = r'/home/zeyu/data/chid_standard/test.txt'
        self.pretrain_model = r"hfl/chinese-roberta-wwm-ext"
        self.save_model_dir = r'/home/zeyu/data/saved_model/idiom_embed/'
        self.output_dir = r'/home/zeyu/data/results/idiom_embed/'
        self.epoch = 20
        self.batch_size = 128
        self.idiom_vocab_size = 3848
        self.bert_lr = 3e-5
        self.lr = 3e-4
        self.warm_up = 0.1
        self.weight_decay = 0.01
        self.dropout = 0.1
        self.bert_hidden_size = 768
        self.idiom_hidden_size = 768
        self.max_length = 160
        self.eval_ratio = 0.1
        self.train_rule = 'Reweight'  # Reweight or DRW or Resample

args = Args()