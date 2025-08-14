from evaluate import load

# تحميل المقاييس
rouge = load("rouge")
accuracy = load("accuracy")
f1 = load("f1")

# ===============================
# دوال مساعدة
# ===============================

def compute_rouge(prediction, reference):
    """
    حساب ROUGE بين نص متوقع ومرجعي
    """
    score = rouge.compute(predictions=[prediction], references=[reference])
    return score

def compute_accuracy(predictions, references):
    """
    حساب Accuracy بين قائمة التوقعات والمرجع
    """
    score = accuracy.compute(predictions=predictions, references=references)
    return score['accuracy']

def compute_f1(predictions, references):
    """
    حساب F1 بين قائمة التوقعات والمرجع
    """
    score = f1.compute(predictions=predictions, references=references)
    return score['f1']
