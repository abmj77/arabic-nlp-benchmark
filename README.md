# 🌍 Arabic NLP Benchmark Suite

## 📚 Overview
The Arabic language remains underserved in NLP despite its complexity and 400M+ speakers, with inconsistent evaluation practices obscuring true model capabilities. We present ArabicBench, the first standardized benchmark for Arabic NLP that evaluates 8 models (including AraGPT2, CAMeL-BERT, and AraT5) across four core tasks—text generation, sentiment analysis, summarization, and question answering—using both classical metrics and Arabic-specific measures. Our framework reveals critical insights: 
(1) task-dependent performance variations (AraT5 excels in summarization but suffers from length bias)
(2) systematic hallucination patterns in generative tasks
(3) significant dialectal performance gaps. 
By establishing reproducible baselines across datasets covering Modern Standard Arabic and three major dialects, this work enables reliable model comparison and drives targeted improvements for Arabic language technologies.


```bash
# 🛠️ Quick Setup (Copy-Paste Ready)
git clone https://github.com/your-repo/arabic-nlp-benchmark.git
cd arabic-nlp-benchmark
python -m venv arabic_nlp_env
source arabic_nlp_env/bin/activate  # Linux/Mac
.\arabic_nlp_env\Scripts\activate  # Windows
pip install -r requirements.txt
```
## 🔍 Core Evaluation Tasks

| Task                | Metrics                          | Supported Models               | Key Challenges                  |
|---------------------|----------------------------------|--------------------------------|---------------------------------|
| 📜 **Text Generation** | BLEU-4, Perplexity, ROUGE-L     | AraGPT2-base	           | Repetition, coherence           |
| 😊 **Sentiment Analysis** | Accuracy, F1, Precision         | CAMeL-BERT-base-mix-sentiment	            | Dialect handling, sarcasm       |
| ✂️ **Text Summarization** | ROUGE-1/2/L, BERTScore         | AraT5-base		                  | Fact consistency, length bias   |
| ❓ **Question Answering** | Exact Match (EM), F1           | AraT5-base	             | Context understanding           |

---

## 🚀 Usage

# Run all benchmark tasks
``` bash
python scripts/run_benchmark.py
```
# Evaluate results and generate metrics report
``` bash
python scripts/evaluate_benchmark.py
```
# Generate interactive report
``` bash
python scripts/visualization/generate_report.py
```
# Clean environment
``` bash
python scripts/clean.py --all
```

---

## 🌟 Key Features
✅ Multi-task evaluation pipeline

✅ Support for 10+ Arabic language models

✅ Automated metric computation

✅ Interactive results visualization

✅ GPU-accelerated inference

---

## Author
Abdulmajeed Abdullah
