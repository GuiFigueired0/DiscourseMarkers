import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# --- 1. CONFIGURATION ---
# We start with the BASE model to establish the baseline
MODEL_ID = "models/roberta-dm-4class-final"
DATASET_ID = "allenai/scitail"
DATASET_CONFIG = "tsv_format"  # SciTail specific format
REPOSITORY_ID = f"models/{MODEL_ID}-finetuned-scitail"
MAX_LENGTH = 512

# --- 2. LOAD DATASET ---
print(f"Loading dataset: {DATASET_ID}")
# SciTail has 'train', 'validation', 'test' splits ready
dataset = load_dataset(DATASET_ID, DATASET_CONFIG)

print(f"Train size: {len(dataset['train'])}")
print(f"Val size:   {len(dataset['validation'])}")
print(f"Test size:  {len(dataset['test'])}")

# --- 3. PROCESS LABELS ---
# SciTail labels: 'entails' vs 'neutral'
print("Encoding dataset labels to ClassLabel...")
dataset = dataset.class_encode_column("label")

class_names = dataset['train'].features['label'].names

num_labels = len(class_names)
id2label = {i: label for i, label in enumerate(class_names)}
label2id = {label: i for i, label in enumerate(class_names)}

print(f"Labels found: {class_names}")

# --- 4. TOKENIZATION ---
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)


def tokenize(batch):
    # Inference tasks usually take two sentences: Premise and Hypothesis
    return tokenizer(
        batch['premise'],  # Sentence 1
        batch['hypothesis'],  # Sentence 2
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )


print("Tokenizing datasets...")
# We use .map to tokenize all splits
tokenized_datasets = dataset.map(tokenize, batched=True)

# Set format for PyTorch
columns_to_keep = ["input_ids", "attention_mask", "label"]
tokenized_datasets.set_format("torch", columns=columns_to_keep)

train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['validation']
test_dataset = tokenized_datasets['test']

# --- 5. METRICS ---
metric_accuracy = evaluate.load("accuracy")
# For 2-class inference, F1 is also useful
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    # Weighted F1 is good for potential imbalance
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


# --- 6. MODEL & TRAINER ---
print("Initializing Model...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.update({"id2label": id2label, "label2id": label2id})

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=config,
    ignore_mismatched_sizes=True
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# SciTail is ~23k train samples.
# 23,000 / 32 = ~718 steps per epoch.
training_args = TrainingArguments(
    output_dir=REPOSITORY_ID,
    num_train_epochs=10,  # Allow room for Early Stopping
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    eval_strategy="steps",
    eval_steps=350,  # Evaluate roughly twice per epoch
    save_strategy="steps",
    save_steps=350,
    logging_strategy="steps",
    logging_steps=350,
    logging_dir=f"{REPOSITORY_ID}/logs",

    learning_rate=2e-5,  # Lower LR for stability
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="accuracy",  # Inference standard is accuracy
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

# --- 7. TRAIN ---
print("Starting Training (Baseline)...")
trainer.train()

# --- 8. EVALUATE & SAVE PREDICTIONS ---
print("Evaluating on Test Set...")
test_results = trainer.predict(test_dataset)

print("Test Metrics:")
print(test_results.metrics)

# Save results to CSV for comparison later
y_pred = np.argmax(test_results.predictions, axis=-1)

df_results = pd.DataFrame({
    'premise': dataset['test']['premise'],
    'hypothesis': dataset['test']['hypothesis'],
    'label': dataset['test']['label'],
    'baseline_pred': y_pred
})

# Map numeric labels back to strings
df_results['label_name'] = df_results['label'].map(id2label)
df_results['baseline_pred_name'] = df_results['baseline_pred'].map(id2label)

csv_filename = 'scitail_experiment_results.csv'
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")