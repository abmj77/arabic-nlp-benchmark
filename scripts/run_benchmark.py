import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import T5ForConditionalGeneration, T5Tokenizer

# === Set up paths ===
datasets_dir = "datasets"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# === Load models ===
# 1. Text Generation
gen_model_name = "aubmindlab/aragpt2-base"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

# 2. Sentiment Analysis
sent_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

# 3. Summarization & QA
t5_model_name = "UBC-NLP/AraT5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# === Functions for each task ===
def run_generation(model, tokenizer, text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=150,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    return scores.argmax().item()

def run_summarization(model, tokenizer, text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs,
        max_length=120,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_qa(model, tokenizer, question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs,
        max_length=100,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Run tasks ===
benchmark_metrics = {}

# 1. Text Generation
gen_results = []
text_gen_path = os.path.join(datasets_dir, "text_generation.csv")
df_gen = pd.read_csv(text_gen_path)
for _, row in df_gen.iterrows():
    try:
        output = run_generation(gen_model, gen_tokenizer, row["input"])
        gen_results.append({"input": row["input"], "output": output})
    except Exception as e:
        gen_results.append({"input": row["input"], "output": f"🚫 {str(e)}"})
pd.DataFrame(gen_results).to_csv(os.path.join(results_dir, "text_generation_results.csv"), index=False)
benchmark_metrics["Text Generation"] = {"Total": len(df_gen), "Errors": sum(1 for r in gen_results if r["output"].startswith("🚫")), "Success": sum(1 for r in gen_results if not r["output"].startswith("🚫"))}

# 2. Sentiment Analysis
sent_results = []
sent_path = os.path.join(datasets_dir, "sentiment.csv")
df_sent = pd.read_csv(sent_path)
for _, row in df_sent.iterrows():
    try:
        output = run_sentiment(sent_model, sent_tokenizer, row["input"])
        sent_results.append({"input": row["input"], "output": output})
    except Exception as e:
        sent_results.append({"input": row["input"], "output": f"🚫 {str(e)}"})
pd.DataFrame(sent_results).to_csv(os.path.join(results_dir, "sentiment_results.csv"), index=False)
benchmark_metrics["Sentiment"] = {"Total": len(df_sent), "Errors": sum(1 for r in sent_results if str(r["output"]).startswith("🚫")), "Success": sum(1 for r in sent_results if not str(r["output"]).startswith("🚫"))}

# 3. Summarization
sum_results = []
sum_path = os.path.join(datasets_dir, "summarization.csv")
df_sum = pd.read_csv(sum_path)
for _, row in df_sum.iterrows():
    try:
        output = run_summarization(t5_model, t5_tokenizer, row["input"])
        sum_results.append({"input": row["input"], "output": output})
    except Exception as e:
        sum_results.append({"input": row["input"], "output": f"🚫 {str(e)}"})
pd.DataFrame(sum_results).to_csv(os.path.join(results_dir, "summarization_results.csv"), index=False)
benchmark_metrics["Summarization"] = {"Total": len(df_sum), "Errors": sum(1 for r in sum_results if str(r["output"]).startswith("🚫")), "Success": sum(1 for r in sum_results if not str(r["output"]).startswith("🚫"))}

# 4. QA
qa_results = []
qa_path = os.path.join(datasets_dir, "qa.csv")
df_qa = pd.read_csv(qa_path)
for _, row in df_qa.iterrows():
    try:
        output = run_qa(t5_model, t5_tokenizer, row["question"], row["context"])
        qa_results.append({"question": row["question"], "context": row["context"], "output": output})
    except Exception as e:
        qa_results.append({"question": row["question"], "context": row["context"], "output": f"🚫 {str(e)}"})
pd.DataFrame(qa_results).to_csv(os.path.join(results_dir, "qa_results.csv"), index=False)
benchmark_metrics["QA"] = {"Total": len(df_qa), "Errors": sum(1 for r in qa_results if str(r["output"]).startswith("🚫")), "Success": sum(1 for r in qa_results if not str(r["output"]).startswith("🚫"))}

# === Create Benchmark Metrics Report ===
report_path = os.path.join(results_dir, "benchmark_metrics.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== Benchmark Metrics Report ===\n\n")
    for task, metrics in benchmark_metrics.items():
        f.write(f"{task}: Total={metrics['Total']}, Errors={metrics['Errors']}, Success={metrics['Success']}\n")

print("✅ Benchmark completed. Results and metrics saved in 'results/' folder.")
