import torch
from transformers import (
    Trainer,
    AutoConfig,
    TrainingArguments,
    RobertaTokenizerFast,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
)

import os
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Custom WeightedLossTrainer Class ---
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- 2. Define Compute Metrics Function ---
metric_f1 = evaluate.load("f1")
metric_accuracy = evaluate.load("accuracy")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1_weighted = metric_f1.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )
    f1_macro = metric_f1.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )
    accuracy = metric_accuracy.compute(
        predictions=predictions,
        references=labels
    )
    precision = metric_precision.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )
    recall = metric_recall.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1_weighted": f1_weighted["f1"],
        "f1_macro": f1_macro["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }


# --- 3. Main Training Logic ---
def main():
    print("Starting Discourse Marker Model Training...")

    # --- Load Data ---
    df = pd.read_csv(os.path.join('data', 'en.csv'))

    # --- Define Full DM Map ---
    dm_to_class_map = {
        # == Contrastive Discourse Markers (CDMs) ==
        # Show opposition, contrast, concession, or correction
        'although': 'CDM',
        'but': 'CDM',
        'by comparison': 'CDM',
        'by contrast': 'CDM',
        'conversely': 'CDM',
        'however': 'CDM',
        'in contrast': 'CDM',
        'instead': 'CDM',
        'nevertheless': 'CDM',
        'nonetheless': 'CDM',
        'on the contrary': 'CDM',
        'on the other hand': 'CDM',
        'otherwise': 'CDM',
        'rather': 'CDM',
        'regardless': 'CDM',
        'still': 'CDM',
        'though': 'CDM',
        'yet': 'CDM',

        # == Elaborative Discourse Markers (EDMs) ==
        # Add info, specify, rephrase, give examples, or add speaker stance
        'absolutely': 'EDM',
        'actually': 'EDM',
        'additionally': 'EDM',
        'admittedly': 'EDM',
        'again': 'EDM',
        'also': 'EDM',
        'alternately': 'EDM',
        'alternatively': 'EDM',
        'altogether': 'EDM',
        'amazingly': 'EDM',
        'and': 'EDM',
        'anyway': 'EDM',
        'apparently': 'EDM',
        'arguably': 'EDM',
        'basically': 'EDM',
        'besides': 'EDM',
        'certainly': 'EDM',
        'clearly': 'EDM',
        'coincidentally': 'EDM',
        'collectively': 'EDM',
        'curiously': 'EDM',
        'elsewhere': 'EDM',
        'especially': 'EDM',
        'essentially': 'EDM',
        'evidently': 'EDM',
        'for example': 'EDM',
        'for instance': 'EDM',
        'fortunately': 'EDM',
        'frankly': 'EDM',
        'further': 'EDM',
        'furthermore': 'EDM',
        'generally': 'EDM',
        'happily': 'EDM',
        'here': 'EDM',
        'honestly': 'EDM',
        'hopefully': 'EDM',
        'ideally': 'EDM',
        'importantly': 'EDM',
        'in fact': 'EDM',
        'in other words': 'EDM',
        'in particular': 'EDM',
        'in short': 'EDM',
        'in sum': 'EDM',
        'incidentally': 'EDM',
        'indeed': 'EDM',
        'interestingly': 'EDM',
        'ironically': 'EDM',
        'likewise': 'EDM',
        'locally': 'EDM',
        'luckily': 'EDM',
        'maybe': 'EDM',
        'meaning': 'EDM',
        'moreover': 'EDM',
        'mostly': 'EDM',
        'namely': 'EDM',
        'nationally': 'EDM',
        'naturally': 'EDM',
        'notably': 'EDM',
        'obviously': 'EDM',
        'oddly': 'EDM',
        'only': 'EDM',
        'optionally': 'EDM',
        'or': 'EDM',
        'overall': 'EDM',
        'particularly': 'EDM',
        'perhaps': 'EDM',
        'personally': 'EDM',
        'plus': 'EDM',
        'preferably': 'EDM',
        'presumably': 'EDM',
        'probably': 'EDM',
        'realistically': 'EDM',
        'really': 'EDM',
        'remarkably': 'EDM',
        'sadly': 'EDM',
        'separately': 'EDM',
        'seriously': 'EDM',
        'significantly': 'EDM',
        'similarly': 'EDM',
        'specifically': 'EDM',
        'strangely': 'EDM',
        'supposedly': 'EDM',
        'surely': 'EDM',
        'surprisingly': 'EDM',
        'technically': 'EDM',
        'thankfully': 'EDM',
        'theoretically': 'EDM',
        'together': 'EDM',
        'truly': 'EDM',
        'truthfully': 'EDM',
        'undoubtedly': 'EDM',
        'unfortunately': 'EDM',
        'unsurprisingly': 'EDM',
        'well': 'EDM',

        # == Implicative Discourse Markers (IDMs) ==
        # Show result, consequence, or inference
        'accordingly': 'IDM',
        'as a result': 'IDM',
        'because of that': 'IDM',
        'because of this': 'IDM',
        'by doing this': 'IDM',
        'consequently': 'IDM',
        'hence': 'IDM',
        'in turn': 'IDM',
        'inevitably': 'IDM',
        'so': 'IDM',
        'thereby': 'IDM',
        'therefore': 'IDM',
        'thus': 'IDM',

        # == Temporal Discourse Markers (TDMs) ==
        # Show time or sequence
        'afterward': 'TDM',
        'already': 'TDM',
        'by then': 'TDM',
        'currently': 'TDM',
        'eventually': 'TDM',
        'finally': 'TDM',
        'first': 'TDM',
        'firstly': 'TDM',
        'frequently': 'TDM',
        'gradually': 'TDM',
        'historically': 'TDM',
        'immediately': 'TDM',
        'in the end': 'TDM',
        'in the meantime': 'TDM',
        'increasingly': 'TDM',
        'initially': 'TDM',
        'lastly': 'TDM',
        'lately': 'TDM',
        'later': 'TDM',
        'meantime': 'TDM',
        'meanwhile': 'TDM',
        'next': 'TDM',
        'normally': 'TDM',
        'now': 'TDM',
        'occasionally': 'TDM',
        'often': 'TDM',
        'once': 'TDM',
        'originally': 'TDM',
        'presently': 'TDM',
        'previously': 'TDM',
        'recently': 'TDM',
        'second': 'TDM',
        'secondly': 'TDM',
        'simultaneously': 'TDM',
        'slowly': 'TDM',
        'sometimes': 'TDM',
        'soon': 'TDM',
        'subsequently': 'TDM',
        'suddenly': 'TDM',
        'then': 'TDM',
        'thereafter': 'TDM',
        'third': 'TDM',
        'thirdly': 'TDM',
        'traditionally': 'TDM',
        'typically': 'TDM',
        'ultimately': 'TDM',
        'usually': 'TDM',
    }

    # --- Process Data ---
    print("Processing dataframe...")
    df['label'] = [dm_to_class_map.get(str(dm).lower().strip()) for dm in df.dm]
    print(f'Original size: {len(df)}')
    df = df.loc[df['label'].notnull()].copy()
    print(f'Size after filtering: {len(df)}')

    dataset = Dataset.from_pandas(df)
    dataset = dataset.class_encode_column("label")

    # --- Stratified Split ---
    print("Creating stratified splits...")
    train_test_split = dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label"
    )
    test_val_split = train_test_split['test'].train_test_split(
        test_size=0.5,
        seed=42,
        stratify_by_column="label"
    )

    train_dataset = train_test_split['train']
    val_dataset = test_val_split['train']
    test_dataset = test_val_split['test']

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # --- Get Labels for Config ---
    class_names = train_dataset.features['label'].names
    num_labels = len(class_names)
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}

    print(f"Number of labels: {num_labels}")
    print(f"The labels: {class_names}")
    print(f"id2label map: {id2label}")

    # --- Tokenize ---
    print("Tokenizing datasets...")
    model_id = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    def tokenize(batch):
        return tokenizer(
            batch['s1'],
            batch['s2'],
            truncation=True,
            max_length=256,
            padding=False
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['s1', 's2', 'dm', 'article_id'])
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=['s1', 's2', 'dm', 'article_id'])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['s1', 's2', 'dm', 'article_id'])

    columns_to_keep = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format("torch", columns=columns_to_keep)
    val_dataset.set_format("torch", columns=columns_to_keep)
    test_dataset.set_format("torch", columns=columns_to_keep)

    # --- Calculate Class Weights ---
    print("Calculating class weights...")

    y_train = np.array(train_dataset['label'])
    known_classes = np.arange(num_labels)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=known_classes,
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Computed weights: {class_weights_tensor}")

    # --- Define Model and Trainer ---
    print("Loading model and config...")
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label, "label2id": label2id})
    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    repository_id = "./models/roberta-base-dm-4class"

    training_args = TrainingArguments(
        output_dir=repository_id,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        logging_strategy="steps",
        logging_steps=2000,
        logging_dir=f"{repository_id}/logs",
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="tensorboard",
        metric_for_best_model="f1_macro",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=class_weights_tensor
    )

    # --- Train ---
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- Save Final Model ---
    final_model_path = "./models/roberta-dm-4class-final"
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("--- Model Saved ---")

    # --- Evaluate on Test Set ---
    print("--- Evaluating on Test Set ---")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(test_results)
    print("--- All Done ---")


# --- Run the main function when the script is executed ---
if __name__ == "__main__":
    main()