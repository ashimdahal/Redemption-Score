import os
import json
from pathlib import Path
import statistics

def print_metrics(model_name):
    metrics_path = Path(f"evaluation_results/{model_name}/metrics.json")
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Extract and format metrics
    bleu = data['Bleu']['bleu'] * 100
    bleu_precisions = [p * 100 for p in data['Bleu']['precisions']]
    
    meteor = data['meteor']['meteor'] * 100
    
    bert_precision = statistics.mean(data['bertscore']['precision']) * 100
    bert_recall = statistics.mean(data['bertscore']['recall']) * 100
    bert_f1 = statistics.mean(data['bertscore']['f1']) * 100
    
    rouge1 = data['rouge']['rouge1'] * 100
    rouge2 = data['rouge']['rouge2'] * 100
    rougeL = data['rouge']['rougeL'] * 100
    rougeL_sum = data['rouge']['rougeLsum'] * 100
    
    google_bleu = data['google_bleu']['google_bleu'] * 100

    # Print formatted output
    print(f"\n{' Metric ':━^40}")
    print(f"BLEU: {bleu:.2f}%")
    for i, p in enumerate(bleu_precisions, 1):
        print(f"BLEU-{i}: {p:.2f}%")
    print(f"METEOR: {meteor:.2f}%")
    print(f"BERTScore Precision: {bert_precision:.2f}%")
    print(f"BERTScore Recall: {bert_recall:.2f}%")
    print(f"BERTScore F1: {bert_f1:.2f}%")
    print(f"ROUGE-1: {rouge1:.2f}%")
    print(f"ROUGE-2: {rouge2:.2f}%")
    print(f"ROUGE-L: {rougeL:.2f}%")
    print(f"ROUGE-L sum: {rougeL_sum:.2f}%")
    print(f"Google BLEU: {google_bleu:.2f}%")
    print("━" * 40)

def main():
    for dir in os.listdir("./evaluation_results/"):
        print(f"{dir}")
        print_metrics(dir)

if __name__ == "__main__":
    main()
