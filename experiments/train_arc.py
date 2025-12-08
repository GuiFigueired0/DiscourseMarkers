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
MODEL_ID = "roberta-base"

# This dataset contains the AAE2 data pre-processed into pairs
DATASET_ID = "DFKI-SLT/cross-domain-argumentation"
DATASET_CONFIG = "student_essays"
REPOSITORY_ID = f"models/roberta-aae2-baseline"
MAX_LENGTH = 512

# --- 2. LOAD DATASET ---
print(f"Loading dataset: {DATASET_ID} ({DATASET_CONFIG})")
dataset = load_dataset(DATASET_ID, DATASET_CONFIG)

# Inspect the dataset to confirm column names
print("\n--- Data Sample ---")
print(dataset['train'][0])
# Expected format: {'argument_1': '...', 'argument_2': '...', 'label': ...}
print("-------------------\n")

print(f"Train size: {len(dataset['train'])}")
print(f"Val size:   {len(dataset['validation'])}")
print(f"Test size:  {len(dataset['test'])}")

# --- 3. PROCESS LABELS ---
# This dataset usually comes with labels like: 0 (Attack), 1 (Support), 2 (No Relation)

# Extract class names automatically
if 'label' in dataset['train'].features:
    features = dataset['train'].features['label']
    if hasattr(features, 'names'):
        class_names = features.names
    else:
        class_names = sorted(list(set(dataset['train']['label'])))
else:
    # Fallback if metadata is missing
    class_names = ["attack", "support", "no_relation"]

num_labels = len(class_names)
id2label = {i: label for i, label in enumerate(class_names)}
label2id = {label: i for i, label in enumerate(class_names)}

print(f"Labels found: {class_names}")

# --- 4. TOKENIZATION ---
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)


def tokenize(batch):
    # This dataset uses 'argument_1' and 'argument_2'
    return tokenizer(
        batch['argument_1'],
        batch['argument_2'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )


print("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize, batched=True)

# Format for PyTorch
columns_to_keep = ["input_ids", "attention_mask", "label"]
tokenized_datasets.set_format("torch", columns=columns_to_keep)

# --- 5. METRICS ---
# F1 Macro is crucial here because 'Attack' relations are often rarer than 'Support'
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)

    # 'macro' gives equal weight to 'Attack' and 'Support' performance
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = metric_f1.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"]
    }


# --- 6. MODEL & TRAINER ---
print("Initializing Model...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.update({"id2label": id2label, "label2id": label2id})

# NOTE: When switching to DM model, add 'ignore_mismatched_sizes=True'
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=config,
    # ignore_mismatched_sizes=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=REPOSITORY_ID,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    # Eval every ~10% of training
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_strategy="steps",
    logging_steps=200,
    logging_dir=f"{REPOSITORY_ID}/logs",

    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=300,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="f1_macro",  # Optimize for the hardest class
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# --- 7. TRAIN ---
print("Starting Training...")
trainer.train()

# --- 8. EVALUATE & SAVE ---
print("Evaluating on Test Set...")
test_results = trainer.predict(tokenized_datasets['test'])

print("Test Metrics:")
print(test_results.metrics)

# Save predictions for comparison
y_pred = np.argmax(test_results.predictions, axis=-1)

df_results = pd.DataFrame({
    'arg1': dataset['test']['argument_1'],
    'arg2': dataset['test']['argument_2'],
    'label': dataset['test']['label'],
    'pred': y_pred
})

# Map predictions to names
df_results['label_name'] = df_results['label'].map(id2label)
df_results['pred_name'] = df_results['pred'].map(id2label)

csv_filename = 'aae2_experiment_results.csv'
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")