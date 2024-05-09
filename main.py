from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = load_dataset("imdb")
tokenized_data = dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)

