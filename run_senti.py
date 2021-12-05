import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from utils.loggers import init_logger, logger
from data.loader import TaskDataSet, TaskBatchSampler, MultiTaskDataset
from data.load_data import load_senti_data
from utils.utils import my_collate, args
from model.sentiment_classifier import MultiTaskModel
from torch import nn
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    init_logger()

    train_data = load_senti_data(args.train_file, "train")
    dev_data = load_senti_data(args.dev_file, "dev")
    test_data = load_senti_data(args.test_file, "test")

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    train_dataset = TaskDataSet("senti", "train", train_data, args.max_length, tokenizer)
    train_dataset = MultiTaskDataset(train_dataset)
    train_sampler = TaskBatchSampler(train_dataset, args.batch_size)
    train_dataLoader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=my_collate
    )

    dev_dataset = TaskDataSet("senti", "dev", dev_data, args.max_length, tokenizer)
    dev_dataset = MultiTaskDataset(dev_dataset)
    dev_sampler = TaskBatchSampler(dev_dataset, args.batch_size)
    dev_dataLoader = DataLoader(
        dataset=dev_dataset,
        sampler=dev_sampler,
        collate_fn=my_collate
    )

    test_dataset = TaskDataSet("senti", "test", test_data, args.max_length, tokenizer)
    test_dataset = MultiTaskDataset(test_dataset)
    test_sampler = TaskBatchSampler(test_dataset, args.batch_size)
    test_dataLoader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        collate_fn=my_collate
    )

    senti_model = MultiTaskModel(args)
    senti_model.to(device)

    # for i, p in enumerate(senti_model.bert.parameters()):
    #     p.requires_grad = False

    fc_params = list()
    fc_params.extend(senti_model.senti_fc.named_parameters())
    optimizer_params = [
        {'params': [p for n, p in fc_params],
         'weight_decay': args.weight_decay, 'lr': args.lr}
    ]

    optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=args.epoch *
                                                len(train_dataLoader) * args.warm_up,
                                                num_training_steps=args.epoch * len(train_dataLoader)
                                                )
    logger.info('start training')
    max_score = -1
    senti_model.train()
    criterion = nn.CrossEntropyLoss()
    scores = []
    losses = []
    epoch_num = train_dataLoader.sampler.get_epoch_num()
    for i in range(args.epoch):
        for step, batch in enumerate(train_dataLoader):
            task_ids = batch[0]
            batch_input_ids = batch[1].to(device)
            batch_token_type_ids = batch[2].to(device)
            batch_attention_mask = batch[3].to(device)
            batch_label_ids = batch[4].to(device)
            prob_logits = senti_model(batch_input_ids, batch_token_type_ids, batch_attention_mask)

            loss = criterion(prob_logits, batch_label_ids)
            logger.info('epoch:{} step:{}/{} loss:{}'.format(i, step, epoch_num, loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(senti_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        score = evaluate(senti_model, dev_dataLoader)
        scores.append(score)

        logger.info("epoch:{} score:{}".format(i, score))
        if score >= max_score:
            max_score = score
            torch.save(senti_model, os.path.join(args.save_model_dir, 'model.pth'))

    with open("data/loss.txt", "w", encoding="utf-8") as wf:
        for loss in losses:
            wf.write(str(loss)+"\n")
    with open("data/score.txt", "w", encoding="utf-8") as wf:
        for score in scores:
            wf.write(str(score)+"\n")

def evaluate(model: MultiTaskModel, dataLoader: DataLoader):
    senti_all_pred = []
    senti_all_target = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataLoader):
            task_id = batch[0]
            batch_input_ids = batch[1].to(device)
            batch_token_type_ids = batch[2].to(device)
            batch_attention_mask = batch[3].to(device)
            batch_label_ids = batch[4].to(device)

            prob_logits = model(batch_input_ids, batch_token_type_ids, batch_attention_mask)
            preds = torch.argmax(prob_logits, dim=1)

            senti_all_target.extend(batch_label_ids.tolist())
            senti_all_pred.extend(preds.tolist())

    score = f1_score(senti_all_target, senti_all_pred, average='macro')
    return score

if __name__ == '__main__':
    train()