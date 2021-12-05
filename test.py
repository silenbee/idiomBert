from transformers import BertTokenizer,BertModel
import json

json.load()

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

text = "我真是寄人篱下"
characters=["寄人篱下"]
tokenizer.add_tokens(characters)

print(tokenizer.tokenize(text))

print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
