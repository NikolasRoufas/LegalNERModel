import tensorflow as tf
from typing import List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer

class LegalDataPreprocessor:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", max_length=128):
        """
        Initialize the data preprocessor.
        
        Args:
            model_name (str): Name of the tokenizer to use
            max_length (int): Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize legal text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def align_labels_with_tokens(self, labels: List[int], word_ids: List[int]) -> List[int]:
        """
        Align labels with tokenized words.
        
        Args:
            labels (List[int]): Original labels
            word_ids (List[int]): Word IDs from tokenizer
            
        Returns:
            List[int]: Aligned labels
        """
        new_labels = []
        current_word = None
        current_label = None
        
        for word_id in word_ids:
            # Special tokens get -100
            if word_id is None:
                new_labels.append(-100)
            # If word_id is different from current_word, it's a new word
            elif word_id != current_word:
                current_word = word_id
                # Handle case where word_id is out of range
                if word_id < len(labels):
                    current_label = labels[word_id]
                    new_labels.append(current_label)
                else:
                    current_label = 0  # Default to 'O' label
                    new_labels.append(current_label)
            # If word_id is the same as current_word, it's a continuation of the same word
            else:
                # For IOB format, if the previous label was B-XXX, this should be I-XXX
                if current_label is not None and current_label > 0 and current_label % 2 == 1:
                    new_labels.append(current_label + 1)  # Convert to I- label
                else:
                    new_labels.append(current_label)
        
        return new_labels
    
    def preprocess_dataset(self, texts: List[str], labels: List[List[int]]) -> Dict:
        """
        Preprocess the dataset for training.
        
        Args:
            texts (List[str]): List of input texts
            labels (List[List[int]]): List of label sequences
            
        Returns:
            Dict: Preprocessed dataset
        """
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Initialize lists for tokenized inputs
        input_ids = []
        attention_masks = []
        token_type_ids = []
        aligned_labels = []
        
        # Process each text and its labels
        for text, text_labels in zip(cleaned_texts, labels):
            # Tokenize text
            tokenized = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf',
                return_offsets_mapping=True
            )
            
            # Get word IDs for alignment
            word_ids = tokenized.word_ids()
            
            # Align labels with tokens
            text_aligned_labels = self.align_labels_with_tokens(text_labels, word_ids)
            
            # Pad or truncate labels to max_length
            padded_labels = np.full(self.max_length, -100, dtype=np.int32)
            padded_labels[:len(text_aligned_labels)] = text_aligned_labels
            
            # Append to lists
            input_ids.append(tokenized['input_ids'].numpy()[0])
            attention_masks.append(tokenized['attention_mask'].numpy()[0])
            token_type_ids.append(tokenized['token_type_ids'].numpy()[0])
            aligned_labels.append(padded_labels)
        
        # Convert to tensors
        return {
            'input_ids': tf.convert_to_tensor(input_ids),
            'attention_mask': tf.convert_to_tensor(attention_masks),
            'token_type_ids': tf.convert_to_tensor(token_type_ids),
            'labels': tf.convert_to_tensor(aligned_labels)
        }
    
    def create_tf_dataset(self, tokenized_inputs: Dict) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from tokenized inputs.
        
        Args:
            tokenized_inputs (Dict): Tokenized inputs
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': tokenized_inputs['input_ids'],
                'attention_mask': tokenized_inputs['attention_mask'],
                'token_type_ids': tokenized_inputs['token_type_ids']
            },
            tokenized_inputs['labels']
        ))
        
        # Shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=1000).batch(32)
        
        return dataset 