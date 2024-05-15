from torch import no_grad
from ingestion import tokenizer
from training import model


def classify(text):
    inputs = tokenizer(text, return_tensors="pt")
    with no_grad():
        logits = model(**inputs).logits

    result_id = logits.argmax().item()
    return model.config.id2label[result_id]
