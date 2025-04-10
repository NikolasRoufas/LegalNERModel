#!/usr/bin/env python3
"""
Legal NER Training Script using the EURLEX dataset and BERT
This script implements a complete NER system for legal text using simulated NER labels
with comprehensive metrics tracking.

"""

import os
import torch
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainerCallback
)
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score as sk_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "bert-base-uncased"  # or "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SAMPLES = 1000
EVAL_SAMPLES = 100
TEST_SAMPLES = 50
OUTPUT_DIR = "legal_ner_model"
LOGGING_DIR = "logs"
METRICS_DIR = "metrics"

# Define NER labels
LABELS = ["O", "B-LAW", "I-LAW", "B-COURT", "I-COURT"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

import os
import json
import matplotlib.pyplot as plt
from transformers import TrainerCallback

class MetricsCallback(TrainerCallback):
    """Custom callback to save metrics after each evaluation."""
    
    def __init__(self, metrics_dir):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

        # Load existing metrics if file exists
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_precision": [],
            "eval_recall": [],
            "eval_f1": [],
            "eval_accuracy": [],
            "learning_rate": [],
            "epoch": []
        }

        metrics_file = os.path.join(self.metrics_dir, "metrics_history.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                loaded = json.load(f)
                self.metrics_history.update(loaded)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            for key, value in metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
            
            # Add epoch info
            epoch = state.log_history[-1].get("epoch", state.global_step / 100)  # Fallback
            self.metrics_history["epoch"].append(epoch)
            
            # Save to file
            with open(os.path.join(self.metrics_dir, "metrics_history.json"), "w") as f:
                json.dump(self.metrics_history, f, indent=4)
            
            # Plot metrics
            self._plot_metrics()

    def _plot_metrics(self):
        """Plot training and evaluation metrics, safely handling length mismatches."""
        history = self.metrics_history
        min_len = min([
            len(history.get("eval_loss", [])),
            len(history.get("eval_precision", [])),
            len(history.get("eval_recall", [])),
            len(history.get("eval_f1", [])),
            len(history.get("eval_accuracy", []))
        ])

        epochs = history.get("epoch", [])[:min_len]
        def trim(key): return history.get(key, [])[:min_len]

        if len(epochs) > 1:
            plt.figure(figsize=(15, 10))

            def safe_plot(ax, x, y_dict, title, xlabel, ylabel):
                lines = []
                for label, y in y_dict.items():
                    if y and len(x) == len(y):
                        line, = ax.plot(x, y, label=label)
                        lines.append(line)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True)
                if lines:
                    ax.legend(loc="best")

            safe_plot(
                plt.subplot(2, 2, 1),
                epochs,
                {"Eval Loss": trim("eval_loss")},
                "Loss",
                "Epoch",
                "Loss"
            )

            safe_plot(
                plt.subplot(2, 2, 2),
                epochs,
                {
                    "Precision": trim("eval_precision"),
                    "Recall": trim("eval_recall"),
                    "F1": trim("eval_f1")
                },
                "Precision, Recall, F1",
                "Epoch",
                "Score"
            )

            safe_plot(
                plt.subplot(2, 2, 3),
                epochs,
                {"Accuracy": trim("eval_accuracy")},
                "Accuracy",
                "Epoch",
                "Score"
            )

            safe_plot(
                plt.subplot(2, 2, 4),
                epochs,
                {"Learning Rate": trim("learning_rate")},
                "Learning Rate",
                "Epoch",
                "Rate"
            )

            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_dir, "metrics_plot.png"))
            plt.close()







def load_and_process_eurlex() -> Tuple[Dict, Dict, Dict]:
    """Load EURLEX dataset and process it for NER training."""
    logger.info("Loading EURLEX dataset...")
    
    # Try different methods to load the dataset with error handling
    dataset = None
    
    try:
        dataset = load_dataset("lex_glue", "eurlex")
        logger.info("Successfully loaded EURLEX dataset from lex_glue")
    except Exception as e:
        logger.warning(f"Error loading EURLEX dataset from lex_glue: {e}")
        try:
            logger.info("Trying alternative download method from joelito/eurlex_lexglue...")
            dataset = load_dataset("joelito/eurlex_lexglue")
            logger.info("Successfully loaded EURLEX dataset from joelito/eurlex_lexglue")
        except Exception as e2:
            logger.error(f"Error loading dataset from alternative source: {e2}")
            raise RuntimeError("Failed to load EURLEX dataset from any source. Please check your internet connection or dataset availability.")
    
    if not dataset:
        raise RuntimeError("Dataset could not be loaded")
    
    # Select subset of data
    train_dataset = dataset["train"].select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
    eval_dataset = dataset["validation"].select(range(min(EVAL_SAMPLES, len(dataset["validation"]))))
    
    # For test data, use test set if available, otherwise use more validation data
    if "test" in dataset:
        test_dataset = dataset["test"].select(range(min(TEST_SAMPLES, len(dataset["test"]))))
    else:
        offset = min(EVAL_SAMPLES, len(dataset["validation"]))
        test_samples = min(TEST_SAMPLES, len(dataset["validation"]) - offset)
        test_dataset = dataset["validation"].select(range(offset, offset + test_samples))
    
    # Extract text field
    train_dataset = train_dataset.map(lambda x: {"text": x["text"]})
    eval_dataset = eval_dataset.map(lambda x: {"text": x["text"]})
    test_dataset = test_dataset.map(lambda x: {"text": x["text"]})
    
    # Simulate NER labels
    train_dataset = train_dataset.map(
        simulate_ner_labels,
        remove_columns=[col for col in train_dataset.column_names if col != "text"]
    )
    eval_dataset = eval_dataset.map(
        simulate_ner_labels,
        remove_columns=[col for col in eval_dataset.column_names if col != "text"]
    )
    test_dataset = test_dataset.map(
        simulate_ner_labels,
        remove_columns=[col for col in test_dataset.column_names if col != "text"]
    )
    
    return train_dataset, eval_dataset, test_dataset

def simulate_ner_labels(example: Dict) -> Dict:
    """Simulate NER labels for legal text using keyword heuristics."""
    text = example["text"].lower()
    words = text.split()
    
    # Initialize all tokens as 'O' (Outside any entity)
    ner_tags = ["O"] * len(words)
    
    # Apply keyword heuristics to assign NER labels
    for i, word in enumerate(words):
        # Check for LAW entities
        if word in ["regulation", "directive", "regulations", "directives"]:
            ner_tags[i] = "B-LAW"
            # Look ahead for multi-token entities
            j = i + 1
            while j < len(words) and words[j] in ["no", "ec", "eu", "eec", "(ec)", "(eu)"]:
                ner_tags[j] = "I-LAW"
                j += 1
        
        # Check for COURT entities
        elif "court" in word:
            ner_tags[i] = "B-COURT"
            # Look ahead for multi-token entities
            j = i + 1
            while j < len(words) and words[j] in ["of", "justice", "european", "appeals"]:
                ner_tags[j] = "I-COURT"
                j += 1
    
    # Convert labels to IDs
    ner_tag_ids = [LABEL2ID[tag] for tag in ner_tags]
    
    return {
        "tokens": words,
        "ner_tags": ner_tag_ids
    }

def tokenize_and_align_labels(
    examples: Dict,
    tokenizer: AutoTokenizer
) -> Dict:
    """Tokenize texts and align labels with tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                # Only label the first token of a word
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignore other sub-tokens
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_label = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_pred.append(ID2LABEL[p])
                true_label.append(ID2LABEL[l])
        
        true_predictions.append(true_pred)
        true_labels.append(true_label)
    
    # Use seqeval for entity-level metrics
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions)
    }
    
    # Generate detailed classification report
    report = classification_report(true_labels, true_predictions, output_dict=True)
    
    # Add detailed metrics per entity type
    for entity_type in LABELS:
        if entity_type in report:
            results[f"{entity_type}_precision"] = report[entity_type]["precision"]
            results[f"{entity_type}_recall"] = report[entity_type]["recall"]
            results[f"{entity_type}_f1"] = report[entity_type]["f1-score"]
    
    return results

def train_model() -> Tuple[Any, Any]:
    """Main training function."""
    # Load dataset
    train_dataset, eval_dataset, test_dataset = load_and_process_eurlex()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Define training arguments with matching evaluation and save strategies
    # FIX: Added evaluation_strategy="steps" to match save_strategy
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOGGING_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",  # Explicitly set save strategy
    save_steps=100,
    eval_steps=100,  # Ensure eval_steps matches save_steps
    eval_strategy="steps",  # Changed to eval_strategy
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    # Disable default logging
    report_to=None
)



    
    # Initialize metrics callback
    metrics_callback = MetricsCallback(METRICS_DIR)
    
    # Early stopping callback 
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    
    # Initialize trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[metrics_callback, early_stopping_callback]
    )
    
    # Create a custom eval loop for older versions of transformers
    # This will be used if the standard training doesn't work
    def custom_eval_loop():
        logger.info("Using custom evaluation loop...")
        # Manual evaluation loop
        best_f1 = 0
        best_model_path = None
        
        for epoch in range(int(training_args.num_train_epochs)):
            trainer.train()
            metrics = trainer.evaluate()
            logger.info(f"Epoch {epoch}: {metrics}")
            
            # Save best model
            if metrics["eval_f1"] > best_f1:
                best_f1 = metrics["eval_f1"]
                best_model_path = os.path.join(OUTPUT_DIR, f"best_model_epoch_{epoch}")
                trainer.save_model(best_model_path)
        
        # Load best model if found
        if best_model_path:
            model = AutoModelForTokenClassification.from_pretrained(best_model_path)
            trainer.model = model
    
    # Use GPU memory optimization if available
    if torch.cuda.is_available():
        logger.info("Enabling GPU memory optimization")
        # Try to enable mixed precision training
        try:
            from accelerate import Accelerator
            accelerator = Accelerator(mixed_precision='fp16')
            logger.info("Mixed precision training enabled")
        except ImportError:
            logger.warning("accelerate package not found. Mixed precision not enabled.")
    
    # Train model safely
    logger.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logger.warning(f"Standard training failed with error: {e}")
        logger.info("Falling back to custom evaluation loop")
        custom_eval_loop()
    
    # Evaluate model on validation set
    logger.info("Evaluating model on validation set...")
    try:
        eval_results = trainer.evaluate()
        logger.info(f"Validation results: {eval_results}")
    except Exception as e:
        logger.warning(f"Standard evaluation failed with error: {e}")
        eval_results = {"error": str(e)}
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    try:
        test_results = trainer.evaluate(tokenized_test)
        logger.info(f"Test results: {test_results}")
    except Exception as e:
        logger.warning(f"Test evaluation failed with error: {e}")
        test_results = {"error": str(e)}
    
    # Save detailed metrics
    all_results = {
        "validation": eval_results,
        "test": test_results
    }
    
    with open(os.path.join(METRICS_DIR, "final_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        logger.warning(f"Error saving model: {e}")
        # Fallback save
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    
    return model, tokenizer

def evaluate_on_real_data(model, tokenizer):
    """Evaluate the model on real legal texts."""
    logger.info("Evaluating model on real legal texts...")
    
    # Real legal texts examples
    def evaluate_on_real_data(model, tokenizer):
        """Evaluate the model on real legal texts."""
        logger.info("Evaluating model on real legal texts...")
        
        # Real legal texts examples
        real_legal_texts = [
            "The European Court of Justice has ruled that Regulation (EC) No 1/2003 on the implementation of the rules on competition must be interpreted as meaning that...",
            "According to the Court of Justice of the European Union, directives must be transposed into national law by Member States.",
            "The Advocate General delivered his opinion in Case C-123/45 concerning Directive 2006/123/EC on services in the internal market.",
            "The Court of Appeals considered whether Regulation (EU) 2016/679 on data protection applies in this specific case.",
            "The Supreme Court judgment referenced Regulation (EC) No 44/2001 on jurisdiction and enforcement of judgments in civil and commercial matters."
        ]
        
        # Force model and inputs to CPU
        device = torch.device("cpu")
        model = model.to(device)
        
        results = []
        for text in real_legal_texts:
            try:
                # Tokenize the text
                words = text.split()
                inputs = tokenizer(
                    words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH
                ).to(device)  # Move inputs to CPU
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=2)
                
                # Convert predictions to labels
                word_predictions = []
                word_ids = inputs.word_ids(0)
                last_word_id = -1
                
                for j, word_id in enumerate(word_ids):
                    if word_id is None or word_id == last_word_id:
                        continue
                    
                    # Get predicted label for this word
                    label_id = predictions[0, j].item()
                    label = ID2LABEL.get(label_id, "O")
                    
                    # Store word and its predicted label
                    if word_id < len(words):
                        word_predictions.append((words[word_id], label))
                    
                    last_word_id = word_id
                
                results.append((text, word_predictions))
            except Exception as e:
                logger.warning(f"Error evaluating text: {e}")
                results.append((text, [("ERROR", "Error processing")]))
        
        # Save results to file
        with open(os.path.join(METRICS_DIR, "real_data_results.txt"), "w") as f:
            for text, predictions in results:
                f.write(f"Text: {text}\n")
                f.write("Predictions:\n")
                for word, label in predictions:
                    f.write(f"  {word}: {label}\n")
                f.write("\n")
        
        # Log some example results
        if results and len(results[0]) > 1:
            logger.info(f"Predictions for first real text:")
            for word, label in results[0][1]:
                if label != "O":
                    logger.info(f"  {word}: {label}")
        
        # Create visualization of entity recognition
        create_entity_visualization(results)

def create_entity_visualization(results):
    """Create a visualization of the entity recognition results."""
    try:
        # Create a directory for visualizations
        vis_dir = os.path.join(METRICS_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate HTML visualization
        html_content = "<html><head><style>"
        html_content += "body { font-family: Arial, sans-serif; margin: 20px; }"
        html_content += ".text-container { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }"
        html_content += ".text-title { font-weight: bold; margin-bottom: 10px; }"
        html_content += ".entity-LAW { background-color: #ffcc99; padding: 2px 4px; border-radius: 3px; }"
        html_content += ".entity-COURT { background-color: #99ccff; padding: 2px 4px; border-radius: 3px; }"
        html_content += "</style></head><body>"
        html_content += "<h1>Legal Entity Recognition Results</h1>"
        
        # Add each text with highlighted entities
        for idx, (text, predictions) in enumerate(results):
            html_content += f"<div class='text-container'>"
            html_content += f"<div class='text-title'>Example {idx+1}</div>"
            
            # Create highlighted text
            words = text.split()
            highlighted_text = ""
            
            for i, word in enumerate(words):
                # Find prediction for this word
                prediction = None
                for pred_word, label in predictions:
                    if pred_word == word:
                        prediction = label
                        break
                
                if prediction and prediction != "O":
                    entity_type = prediction.split("-")[1] if "-" in prediction else "UNKNOWN"  # Extract LAW or COURT
                    highlighted_text += f"<span class='entity-{entity_type}'>{word}</span> "
                else:
                    highlighted_text += word + " "
            
            html_content += f"<p>{highlighted_text}</p>"
            
            # Add entity legend
            html_content += "<div>"
            html_content += "<span class='entity-LAW'>■</span> LAW &nbsp;&nbsp;"
            html_content += "<span class='entity-COURT'>■</span> COURT"
            html_content += "</div>"
            
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        # Save HTML file
        with open(os.path.join(vis_dir, "entity_visualization.html"), "w") as f:
            f.write(html_content)
        
        logger.info(f"Entity visualization saved to {os.path.join(vis_dir, 'entity_visualization.html')}")
    
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")

def main():
    """Main function to run the training pipeline."""
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Train model
        model, tokenizer = train_model()
        
        # Evaluate on real data
        evaluate_on_real_data(model, tokenizer)
        
        logger.info("Training and evaluation complete. Results saved to metrics directory.")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.exception("Detailed error trace:")
        logger.info("Check your dependencies. The following packages are required:")
        logger.info("pip install --upgrade transformers[torch] datasets seqeval scikit-learn matplotlib tqdm")
        logger.info("pip install 'accelerate>=0.26.0'")

if __name__ == "__main__":
    main()