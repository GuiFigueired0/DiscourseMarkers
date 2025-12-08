import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# --- 1. CONFIGURAÇÃO ---
PATH_MODELO_BASELINE = "./models/roberta-base-anli/final"
PATH_MODELO_DM = "./models/roberta-dm-anli/checkpoint-50900"

DATASET_ID = "facebook/anli"
MAX_LENGTH = 256

# --- 2. PREPARAÇÃO DO DATASET ---
print(f"Loading dataset: {DATASET_ID}")
raw_dataset = load_dataset(DATASET_ID)

# Definir os splits de teste separadamente para comparação com o Leaderboard
test_splits = {
    "Round 1 (A1)": raw_dataset['test_r1'],
    "Round 2 (A2)": raw_dataset['test_r2'],
    "Round 3 (A3)": raw_dataset['test_r3']
}

# --- 3. CONFIGURAÇÃO DE MÉTRICAS ---
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = metric_accuracy.compute(predictions=predictions, references=labels)
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "f1_macro": f1_macro["f1"]
    }


# --- 4. FUNÇÃO DE AVALIAÇÃO ---
def evaluate_model(model_path, model_name, test_splits):
    print(f"\n--- Avaliando Modelo: {model_name} ---")
    print(f"Carregando de: {model_path}")

    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível carregar {model_name}.")
        print(e)
        return {}

    # Tokenização
    def tokenize(batch):
        return tokenizer(
            batch['premise'],
            batch['hypothesis'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./temp", per_device_eval_batch_size=32, report_to="none"),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    results = {}

    # Loop por cada rodada (R1, R2, R3)
    for split_name, dataset in test_splits.items():
        print(f"  > Processando {split_name} ({len(dataset)} amostras)...")
        tokenized_split = dataset.map(tokenize, batched=True)
        # Garantir formato correto
        tokenized_split.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Avaliar
        eval_result = trainer.evaluate(tokenized_split)

        # Salvar apenas o que interessa
        results[split_name] = {
            "Accuracy": eval_result["eval_accuracy"],
            "F1 Macro": eval_result["eval_f1_macro"]
        }
        print(f"    -> Acc: {eval_result['eval_accuracy']:.4f}")

    return results


# --- 5. EXECUÇÃO ---

# Avaliar Baseline
results_baseline = evaluate_model(PATH_MODELO_BASELINE, "Baseline (RoBERTa)", test_splits)

# Avaliar Modelo DM
results_dm = evaluate_model(PATH_MODELO_DM, "Ours (DM Pre-trained)", test_splits)

# --- 6. TABELA COMPARATIVA FINAL ---
print("\n" + "=" * 60)
print("COMPARAÇÃO FINAL COM LEADERBOARD (ACURÁCIA)")
print("=" * 60)
print(f"{'Split / Round':<20} | {'Baseline Acc':<15} | {'DM Model Acc':<15} | {'Difference':<10}")
print("-" * 60)

splits = ["Round 1 (A1)", "Round 2 (A2)", "Round 3 (A3)"]

for split in splits:
    # Obter acurácias (se o modelo falhou, usa 0.0)
    acc_base = results_baseline.get(split, {}).get("Accuracy", 0.0)
    acc_dm = results_dm.get(split, {}).get("Accuracy", 0.0)
    diff = acc_dm - acc_base

    # Formatação visual
    diff_str = f"{diff:+.4f}"

    print(f"{split:<20} | {acc_base:.4f}          | {acc_dm:.4f}          | {diff_str}")

print("=" * 60)

# Salvar resultados completos em CSV para sua tese
data = []
for split in splits:
    row = {"Round": split}
    # Dados Baseline
    row["Baseline_Acc"] = results_baseline.get(split, {}).get("Accuracy")
    row["Baseline_F1"] = results_baseline.get(split, {}).get("F1 Macro")
    # Dados DM
    row["DM_Acc"] = results_dm.get(split, {}).get("Accuracy")
    row["DM_F1"] = results_dm.get(split, {}).get("F1 Macro")
    data.append(row)

df_results = pd.DataFrame(data)
df_results.to_csv("anli_comparison_results.csv", index=False)
print("\nResultados detalhados salvos em 'anli_comparison_results.csv'")