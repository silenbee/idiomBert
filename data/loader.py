import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler

senti = {
    '0': 0,
    '1': 1
}

class TaskDataSet(Dataset):
    def __init__(self, task_name, type, data, maxLength, tokenizer):
        super(TaskDataSet, self).__init__()
        self.task_name = task_name
        self.type = type
        self.data = data
        self.maxLen = maxLength
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self.type == 'train':
            s, label = self.data[index]
        else:
            s, label = self.data[index]
        results = self.tokenizer(
            s,
            max_length=self.maxLen,
            padding='max_length',
            return_tensors='pt',
            truncation=True)

        label_id = senti[label]
        return {'task': self.task_name,
                'data': (results['input_ids'], results['token_type_ids'],
                         results['attention_mask'], torch.LongTensor([label_id]))}

    def __len__(self):
        return len(self.data)

class IdiomChoiceTaskDataSet(Dataset):
    def __init__(self, task_name, type, data, maxLength, tokenizer):
        super(IdiomChoiceTaskDataSet, self).__init__()
        self.task_name = task_name
        self.type = type
        self.data = data
        self.maxLen = maxLength
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self.type == 'train' or self.type == "dev":
            s, candidates, label = self.data[index]
        else:
            s, candidates  = self.data[index]
        results = self.tokenizer(
            s,
            max_length=self.maxLen,
            padding='max_length',
            return_tensors='pt',
            truncation=True)

        mask_position = -1
        for ind, data in enumerate(results['input_ids'][0]):
            if data == 103:
                mask_position = ind

        assert mask_position != -1

        return {'task': self.task_name,
                'data': (results['input_ids'], results['token_type_ids'],
                         results['attention_mask'], torch.LongTensor(candidates),
                         torch.LongTensor([mask_position]), torch.LongTensor([label]))}

    def __len__(self):
        return len(self.data)


class MultiTaskDataset(Dataset):
    def __init__(self, dataset):
        super(MultiTaskDataset, self).__init__()
        self.multi_dataset = {
            0: dataset
        }

    def __getitem__(self, index):
        return [self.multi_dataset[0][idx] for idx in index]

    def get_data_len_by_task_id(self, task_id):
        return len(self.multi_dataset[task_id])


class TaskBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = dataset.get_data_len_by_task_id(0)
        self.task_indices = list(range(0, self.dataset_len))
        if self.shuffle:
            random.shuffle(self.task_indices)
        self.epoch_num = int((self.dataset_len - 1)/self.batch_size) + 1

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        self.all_indices = []
        for epoch_idx in range(self.epoch_num):
            start_ind = epoch_idx * self.batch_size
            end_ind = min((epoch_idx+1) * self.batch_size, self.dataset_len)
            epoch_indices = [self.task_indices[index] for index in range(start_ind, end_ind)]
            self.all_indices.append(epoch_indices)
        for batch_indices in self.all_indices:
            yield batch_indices

    def get_epoch_num(self):
        return self.epoch_num



