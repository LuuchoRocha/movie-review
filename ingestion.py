from transformers import DataCollatorWithPadding, DistilBertTokenizerFast

from datasets import load_dataset

tokenizer = DistilBertTokenizerFast.from_pretrained("./hungry-bert")
dataset = load_dataset("csv", data_files="datasets/comidas/train.csv")
data = dataset.filter(lambda v: len(str(v['text'])) > 5)
data = data.map(lambda value: tokenizer(value["text"], truncation=True), batched=True)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
