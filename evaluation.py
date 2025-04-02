import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class LegalNEREvaluator:
    def __init__(self, id2label: Dict[int, str]):
        """
        Initialize the evaluator.
        
        Args:
            id2label (Dict[int, str]): Mapping from label IDs to label names
        """
        self.id2label = id2label
        self.label_list = list(id2label.values())
        
    def compute_metrics(self, predictions: np.ndarray, 
                       true_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute precision, recall, and F1-score for each entity type.
        
        Args:
            predictions (np.ndarray): Model predictions
            true_labels (np.ndarray): Ground truth labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Flatten predictions and labels
        predictions = predictions.flatten()
        true_labels = true_labels.flatten()
        
        # Remove padding tokens (-100)
        mask = true_labels != -100
        predictions = predictions[mask]
        true_labels = true_labels[mask]
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_confusion_matrix(self, predictions: np.ndarray, 
                            true_labels: np.ndarray,
                            save_path: str = None):
        """
        Plot confusion matrix for entity predictions.
        
        Args:
            predictions (np.ndarray): Model predictions
            true_labels (np.ndarray): Ground truth labels
            save_path (str): Path to save the plot
        """
        # Flatten predictions and labels
        predictions = predictions.flatten()
        true_labels = true_labels.flatten()
        
        # Remove padding tokens (-100)
        mask = true_labels != -100
        predictions = predictions[mask]
        true_labels = true_labels[mask]
        
        # Get unique labels present in the data
        unique_labels = np.unique(np.concatenate([true_labels, predictions]))
        unique_labels = unique_labels[unique_labels != -100]  # Remove -100 if present
        
        # Create label names for the confusion matrix
        label_names = [self.id2label.get(label, f"LABEL_{label}") for label in unique_labels]
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names,
                    yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def print_entity_statistics(self, predictions: np.ndarray, 
                              true_labels: np.ndarray):
        """
        Print statistics for each entity type.
        
        Args:
            predictions (np.ndarray): Model predictions
            true_labels (np.ndarray): Ground truth labels
        """
        # Flatten predictions and labels
        predictions = predictions.flatten()
        true_labels = true_labels.flatten()
        
        # Remove padding tokens (-100)
        mask = true_labels != -100
        predictions = predictions[mask]
        true_labels = true_labels[mask]
        
        # Get unique labels present in the data
        unique_labels = np.unique(np.concatenate([true_labels, predictions]))
        unique_labels = unique_labels[unique_labels != -100]  # Remove -100 if present
        
        print("\nEntity Statistics:")
        print("-" * 50)
        print(f"{'Entity':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        # Compute metrics for each entity type
        for label in unique_labels:
            if label in self.id2label:
                entity_name = self.id2label[label]
                # Create binary masks for this entity
                pred_mask = predictions == label
                true_mask = true_labels == label
                
                # Compute metrics
                if np.any(pred_mask) or np.any(true_mask):
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        true_mask, pred_mask, average='binary', zero_division=0
                    )
                    print(f"{entity_name:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}")
                else:
                    print(f"{entity_name:<15} 0.0000     0.0000     0.0000") 