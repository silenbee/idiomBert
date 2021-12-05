import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from utils.loggers import init_logger, logger
from data.loader import IdiomChoiceTaskDataSet, TaskBatchSampler, MultiTaskDataset
from data.load_data import load_chid_data
from utils.utils import chid_collate, args
from model.idiom_selector import MultiTaskModel
from torch import nn
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    init_logger()

    train_data = load_chid_data(args.train_file)
    dev_data = load_chid_data(args.dev_file)
    test_data = load_chid_data(args.test_file)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    train_dataset = IdiomChoiceTaskDataSet("idiom_embed", "train", train_data, args.max_length, tokenizer)
    train_dataset = MultiTaskDataset(train_dataset)
    train_sampler = TaskBatchSampler(train_dataset, args.batch_size)
    train_dataLoader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=chid_collate,
        num_workers=4
    )

    dev_dataset = IdiomChoiceTaskDataSet("idiom_embed", "dev", dev_data, args.max_length, tokenizer)
    dev_dataset = MultiTaskDataset(dev_dataset)
    dev_sampler = TaskBatchSampler(dev_dataset, args.batch_size)
    dev_dataLoader = DataLoader(
        dataset=dev_dataset,
        sampler=dev_sampler,
        collate_fn=chid_collate,
        num_workers=4
    )

    test_dataset = IdiomChoiceTaskDataSet("idiom_embed", "test", test_data, args.max_length, tokenizer)
    test_dataset = MultiTaskDataset(test_dataset)
    test_sampler = TaskBatchSampler(test_dataset, args.batch_size)
    test_dataLoader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        collate_fn=chid_collate
    )

    idiom_model = MultiTaskModel(args)
    idiom_model.to(device)

    trained_params = list()
    # fix bert
    # for i, p in enumerate(idiom_model.bert.parameters()):
    #     p.requires_grad = False

    for name, param in idiom_model.named_parameters():
        if param.requires_grad:
            trained_params.append((name, param))

    optimizer_params = [
        {'params': [p for n, p in trained_params],
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
    idiom_model.train()
    scores = []
    epoch_num = train_dataLoader.sampler.get_epoch_num()

    for i in range(args.epoch):
        losses = []
        for step, batch in enumerate(train_dataLoader):
            task_ids = batch[0]
            batch_input_ids = batch[1].to(device)
            batch_token_type_ids = batch[2].to(device)
            batch_attention_mask = batch[3].to(device)
            batch_candidates = batch[4].to(device)
            batch_positions = batch[5].to(device)
            batch_label_ids = batch[6].to(device)
            loss, whole_loss = idiom_model(batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_positions, batch_candidates, batch_label_ids)
            medi_loss = 0.8 * loss + 0.2 * whole_loss
            logger.info('epoch:{} step:{}/{} medi_loss:{} loss:{} over loss:{}'.format(i, step, epoch_num, medi_loss, loss, whole_loss))
            medi_loss.backward()
            torch.nn.utils.clip_grad_norm_(idiom_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(medi_loss.item())

            if step != 0 and step % 1000 == 0:
                score = evaluate(idiom_model, dev_dataLoader)
                logger.info("epoch:{} step:{} score:{}".format(i, step, score))
                with open("/home/zeyu/data/results/idiom_embed/score.txt", "a", encoding="utf-8") as wf:
                    wf.write(str(score) + "\n")
                if score >= max_score:
                    max_score = score
                    torch.save(idiom_model, os.path.join(args.save_model_dir, 'idiom_model.pth'))
                    emb_np = np.array(idiom_model.idiom_embedding.weight.data.cpu())
                    np.save(os.path.join(args.save_model_dir, 'idiom_embedding.npy'), emb_np)

        avg_loss = np.mean(losses)
        # write to file
        with open("/home/zeyu/data/results/idiom_embed/loss.txt", "a", encoding="utf-8") as wf:
            wf.write(str(avg_loss) + "\n")

        score = evaluate(idiom_model, dev_dataLoader)
        #write to file
        with open("/home/zeyu/data/results/idiom_embed/score.txt", "a", encoding="utf-8") as wf:
            wf.write(str(score) + "\n")
        scores.append(score)

        logger.info("epoch:{} score:{} avg_loss:{}".format(i, score, avg_loss))
        if score >= max_score:
            max_score = score
            torch.save(idiom_model, os.path.join(args.save_model_dir, 'idiom_model.pth'))
            emb_np = np.array(idiom_model.idiom_embedding.weight.data.cpu())
            np.save(os.path.join(args.save_model_dir, 'idiom_embedding.npy'), emb_np)


def evaluate(model: MultiTaskModel, dataLoader: DataLoader):
    idiom_all_pred = []
    idiom_all_target = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataLoader):
            task_id = batch[0]
            batch_input_ids = batch[1].to(device)
            batch_token_type_ids = batch[2].to(device)
            batch_attention_mask = batch[3].to(device)
            batch_candidates = batch[4].to(device)
            batch_positions = batch[5].to(device)
            batch_label_ids = batch[6].to(device)

            output_logits, wo, co = model(batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_positions, batch_candidates)
            preds = torch.argmax(output_logits, dim=1)

            idiom_all_target.extend(batch_label_ids.tolist())
            idiom_all_pred.extend(preds.tolist())

    score = f1_score(idiom_all_target, idiom_all_pred, average='macro')
    return score

if __name__ == '__main__':
    train()