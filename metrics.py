import evaluate
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)
