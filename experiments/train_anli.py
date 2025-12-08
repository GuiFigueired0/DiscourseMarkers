import torch
import numpy as np
import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

MODEL_ID = "models/roberta-dm-4class-final"
DATASET_ID = "facebook/anli"
REPOSITORY_ID = "models/roberta-dm-anli"
MAX_LENGTH = 256

print(f"Loading dataset: {DATASET_ID}")
raw_dataset = load_dataset(DATASET_ID)

train_dataset = concatenate_datasets([
    raw_dataset['train_r1'],
    raw_dataset['train_r2'],
    raw_dataset['train_r3']
])

eval_dataset = concatenate_datasets([
    raw_dataset['dev_r1'],
    raw_dataset['dev_r2'],
    raw_dataset['dev_r3']
])

test_dataset = concatenate_datasets([
    raw_dataset['test_r1'],
    raw_dataset['test_r2'],
    raw_dataset['test_r3']
])

print(f"Total Train size: {len(train_dataset)}")
print(f"Total Eval size:  {len(eval_dataset)}")
print(f"Total Test size:  {len(test_dataset)}")

if 'label' in train_dataset.features:
    class_names = train_dataset.features['label'].names
    num_labels = len(class_names)
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}
else:
    class_names = ["Entailment", "Neutral", "Contradiction"]
    num_labels = 3
    id2label = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    label2id = {"Entailment": 0, "Neutral": 1, "Contradiction": 2}

print(f"Labels: {class_names}")

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)

def tokenize(batch):
    return tokenizer(
        batch['premise'],
        batch['hypothesis'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False  # DataCollator will handle padding dynamically
    )


print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_eval = eval_dataset.map(tokenize, batched=True)
tokenized_test = test_dataset.map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
tokenized_train.set_format("torch", columns=cols)
tokenized_eval.set_format("torch", columns=cols)
tokenized_test.set_format("torch", columns=cols)

metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = metric_f1.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"]
    }



print("Initializing Model...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.update({"id2label": id2label, "label2id": label2id, "num_labels": num_labels})

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=config,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir=REPOSITORY_ID,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    eval_strategy="steps",
    eval_steps=15000,
    save_strategy="steps",
    save_steps=15000,
    logging_strategy="steps",
    logging_steps=15000,

    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=10000,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Starting Training...")
trainer.train()

print("Evaluating on Test Set...")
results = trainer.evaluate(tokenized_test)
print("Test Results:", results)

trainer.save_model(f"./{REPOSITORY_ID}/final")
tokenizer.save_pretrained(f"./{REPOSITORY_ID}/final")
print("Model saved.")