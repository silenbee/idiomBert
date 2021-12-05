import os
import torch
from transformers import pipeline, BertForSequenceClassification, AutoTokenizer

# model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

result = tokenizer("我是一个中国人", truncation=True)
print(result)

result = tokenizer("我是谁", truncation=True)
print(result)

# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#
# print(classifier("我很开心"))
# print(classifier("我很伤心"))