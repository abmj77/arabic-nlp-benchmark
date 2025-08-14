from evaluate import load

# Load metrics
rouge = load("rouge")
accuracy = load("accuracy")
f1 = load("f1")

# ===============================
# Helper functions
# ===============================

def compute_rouge(prediction, reference):
    """
    Compute ROUGE between a predicted and reference text
    """
    score = rouge.compute(predictions=[prediction], references=[reference])
    return score

def compute_accuracy(predictions, references):
    """
    Compute Accuracy between a list of predictions and references
    """
    score = accuracy.compute(predictions=predictions, references=references)
    return score['accuracy']

def compute_f1(predictions, references):
    """
    Compute F1 between a list of predictions and references
    """
    score = f1.compute(predictions=predictions, references=references)
    return score['f1']
