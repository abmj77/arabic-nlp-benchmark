# evaluate_benchmark.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer

def evaluate_text_generation(file_path):
    df = pd.read_csv(file_path)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []

    for _, row in df.iterrows():
        try:
            ref = str(row.get('output', ''))
            pred = str(row.get('predicted', row.get('output', '')))
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        except Exception:
            continue

    return {
        'ROUGE-1': sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0,
        'ROUGE-L': sum(rougeL_scores)/len(rougeL_scores) if rougeL_scores else 0,
        'Total': len(df),
        'Evaluated': len(rouge1_scores)
    }

def evaluate_summarization(file_path):
    df = pd.read_csv(file_path)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []

    for _, row in df.iterrows():
        try:
            ref = str(row.get('output', ''))
            pred = str(row.get('predicted', row.get('output', '')))
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        except Exception:
            continue

    return {
        'ROUGE-1': sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0,
        'ROUGE-L': sum(rougeL_scores)/len(rougeL_scores) if rougeL_scores else 0,
        'Total': len(df),
        'Evaluated': len(rouge1_scores)
    }

def evaluate_sentiment(file_path):
    df = pd.read_csv(file_path)
    # Use 'predicted' if available, otherwise use 'output'
    y_true = df['output'].astype(int)
    if 'predicted' in df.columns:
        y_pred = df['predicted'].astype(int)
    else:
        y_pred = df['output'].astype(int)

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, average='macro'),
        'Total': len(df)
    }

def evaluate_qa(file_path):
    df = pd.read_csv(file_path)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []

    for _, row in df.iterrows():
        try:
            ref = str(row.get('output', ''))
            pred = str(row.get('predicted', row.get('output', '')))
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        except Exception:
            continue

    return {
        'ROUGE-1': sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0,
        'ROUGE-L': sum(rougeL_scores)/len(rougeL_scores) if rougeL_scores else 0,
        'Total': len(df),
        'Evaluated': len(rouge1_scores)
    }

if __name__ == "__main__":
    print("=== Evaluating Arabic NLP Benchmark ===\n")

    print("Text Generation Results:")
    print(evaluate_text_generation("results/text_generation_results.csv"))

    print("\nSummarization Results:")
    print(evaluate_summarization("results/summarization_results.csv"))

    print("\nSentiment Results:")
    print(evaluate_sentiment("results/sentiment_results.csv"))

    print("\nQA Results:")
    print(evaluate_qa("results/qa_results.csv"))
