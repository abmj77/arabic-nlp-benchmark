# scripts/generate_report.py
import pandas as pd
import os
from evaluate_benchmark import evaluate_text_generation, evaluate_summarization, evaluate_sentiment, evaluate_qa

# تأكد أن مجلد النتائج موجود
os.makedirs("results", exist_ok=True)

report_data = []

# تقييم كل مهمة
print("Evaluating Text Generation...")
text_gen_metrics = evaluate_text_generation("results/text_generation_results.csv")
report_data.append({"Task": "Text Generation", **text_gen_metrics})

print("Evaluating Summarization...")
summarization_metrics = evaluate_summarization("results/summarization_results.csv")
report_data.append({"Task": "Summarization", **summarization_metrics})

print("Evaluating Sentiment...")
sentiment_metrics = evaluate_sentiment("results/sentiment_results.csv")
report_data.append({"Task": "Sentiment", **sentiment_metrics})

print("Evaluating QA...")
qa_metrics = evaluate_qa("results/qa_results.csv")
report_data.append({"Task": "QA", **qa_metrics})

# تحويل البيانات إلى DataFrame
df_report = pd.DataFrame(report_data)

# حفظ التقرير كـ CSV و TXT
df_report.to_csv("results/benchmark_report.csv", index=False)
with open("results/benchmark_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Benchmark Metrics Report ===\n\n")
    for _, row in df_report.iterrows():
        f.write(f"{row['Task']} Metrics:\n")
        for key, value in row.items():
            if key != "Task":
                f.write(f"{key}: {value}\n")
        f.write("\n")

print("✅ Benchmark report generated in 'results/benchmark_report.csv' and 'results/benchmark_report.txt'.")
