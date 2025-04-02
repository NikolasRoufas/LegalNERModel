import tensorflow as tf
import numpy as np
import pandas as pd
from legal_ner_model import LegalNERModel
from data_preprocessing import LegalDataPreprocessor
from evaluation import LegalNEREvaluator

def load_data(csv_path):
    """Load data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Convert labels from string to list of integers
    texts = df['text'].tolist()
    labels = [list(map(int, label_str.split(','))) for label_str in df['labels']]
    
    return texts, labels

def main():
    # Example label mapping (customize based on your dataset)
    id2label = {
        0: "O",  # Outside
        1: "B-PERSON",
        2: "I-PERSON",
        3: "B-ORGANIZATION",
        4: "I-ORGANIZATION",
        5: "B-LOCATION",
        6: "I-LOCATION",
        7: "B-DATE",
        8: "I-DATE"
    }
    
    # Initialize components
    model = LegalNERModel(num_labels=len(id2label))
    preprocessor = LegalDataPreprocessor()
    evaluator = LegalNEREvaluator(id2label)
    
    # Load data from CSV
    texts, labels = load_data('legal_ner_data.csv')
    
    # Preprocess data
    tokenized_inputs = preprocessor.preprocess_dataset(texts, labels)
    
    # Split the data before creating the dataset
    total_samples = len(texts)
    val_size = 2
    train_size = total_samples - val_size
    
    # Split the inputs
    train_inputs = {
        'input_ids': tokenized_inputs['input_ids'][:train_size],
        'attention_mask': tokenized_inputs['attention_mask'][:train_size],
        'token_type_ids': tokenized_inputs['token_type_ids'][:train_size],
        'labels': tokenized_inputs['labels'][:train_size]
    }
    
    val_inputs = {
        'input_ids': tokenized_inputs['input_ids'][train_size:],
        'attention_mask': tokenized_inputs['attention_mask'][train_size:],
        'token_type_ids': tokenized_inputs['token_type_ids'][train_size:],
        'labels': tokenized_inputs['labels'][train_size:]
    }
    
    # Create datasets
    train_dataset = preprocessor.create_tf_dataset(train_inputs)
    val_dataset = preprocessor.create_tf_dataset(val_inputs)
    
    # Print dataset sizes
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Compile and train model
    model.compile_model()
    model.train(train_dataset, val_dataset, epochs=3)
    
    # Evaluate model
    predictions = model.model.predict(val_dataset)
    predictions = np.argmax(predictions.logits, axis=-1)
    
    # Get true labels
    true_labels = []
    for _, labels in val_dataset:
        true_labels.append(labels.numpy())
    true_labels = np.concatenate(true_labels)
    
    # Compute and print metrics
    metrics = evaluator.compute_metrics(predictions, true_labels)
    print("\nOverall Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Print entity-specific statistics
    evaluator.print_entity_statistics(predictions, true_labels)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(predictions, true_labels, "confusion_matrix.png")
    
    # Example prediction
    test_text = "The Supreme Court will hear the case on March 1, 2024."
    entities = model.predict(test_text)
    
    print("\nExample Prediction:")
    print(f"Text: {test_text}")
    print("Entities:")
    for entity in entities:
        print(f"- {entity['text']} ({id2label[entity['label']]})")

if __name__ == "__main__":
    main() 