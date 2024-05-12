from transformers import DataCollatorWithPadding, DistilBertTokenizerFast

from datasets import load_from_disk

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_from_disk("./datasets/imdb")
data = dataset.map(lambda value: tokenizer(value["text"], truncation=True), batched=True)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
