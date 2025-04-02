import tensorflow as tf
from transformers import TFBertForTokenClassification, AutoConfig, AutoTokenizer
from typing import List, Dict, Tuple
import numpy as np

class LegalNERModel:
    def __init__(self, num_labels, model_name="nlpaueb/legal-bert-base-uncased"):
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.compile_model()

    def compile_model(self):
        """Compile the model with appropriate loss and metrics."""
        # Load model configuration
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label={i: f"LABEL_{i}" for i in range(self.num_labels)},
            label2id={f"LABEL_{i}": i for i in range(self.num_labels)}
        )
        
        # Initialize model
        self.model = TFBertForTokenClassification.from_pretrained(
            self.model_name,
            config=config
        )
        
        # Define class weights to handle imbalance
        # Higher weights for entity classes (1-8) and lower weight for 'O' class (0)
        class_weights = tf.constant([0.5] + [2.0] * (self.num_labels - 1), dtype=tf.float32)
        
        # Define custom loss function that handles -100 labels and class imbalance
        def masked_sparse_categorical_crossentropy(y_true, y_pred):
            # Create a mask for valid labels (not -100)
            mask = tf.cast(y_true != -100, dtype=tf.float32)
            
            # Convert labels to one-hot encoding
            y_true_one_hot = tf.one_hot(y_true, self.num_labels)
            
            # Compute loss for all positions
            loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred, from_logits=True)
            
            # Apply class weights only to valid labels
            valid_labels = tf.where(y_true != -100, y_true, 0)  # Replace -100 with 0 for weight lookup
            weights = tf.gather(class_weights, valid_labels)
            weighted_loss = loss * weights
            
            # Apply mask and compute mean loss over non-padding tokens
            masked_loss = weighted_loss * mask
            return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        
        # Define custom accuracy metric that handles -100 labels
        class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
            def __init__(self, name='masked_sparse_categorical_accuracy', **kwargs):
                super().__init__(name=name, **kwargs)
                self.total = self.add_weight(name='total', initializer='zeros')
                self.count = self.add_weight(name='count', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                # Create mask for valid labels
                mask = tf.cast(y_true != -100, dtype=tf.float32)
                
                # Get predictions and ensure same type as y_true
                y_pred = tf.argmax(y_pred, axis=-1)
                y_pred = tf.cast(y_pred, dtype=y_true.dtype)
                
                # Compute accuracy only for valid tokens
                correct = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
                masked_correct = correct * mask
                
                self.total.assign_add(tf.reduce_sum(masked_correct))
                self.count.assign_add(tf.reduce_sum(mask))

            def result(self):
                return self.total / self.count

            def reset_state(self):
                self.total.assign(0)
                self.count.assign(0)
        
        # Compile model with custom loss and metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-5),
            loss=masked_sparse_categorical_crossentropy,
            metrics=[MaskedSparseCategoricalAccuracy()]
        )

    def train(self, train_dataset, val_dataset, epochs=3):
        """Train the model."""
        # Check if datasets are empty
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Training or validation dataset is empty")
        
        # Train the model with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=1
            )
        ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def predict(self, text):
        """Make predictions on new text."""
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get predictions
        outputs = self.model(inputs)
        predictions = tf.argmax(outputs.logits, axis=-1)
        
        # Convert predictions to entities
        entities = []
        current_entity = None
        current_text = ""
        
        for i, (token, pred) in enumerate(zip(inputs["input_ids"][0], predictions[0])):
            # Skip special tokens
            if token in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                continue
                
            # Get the word
            word = self.tokenizer.decode([token])
            
            # Handle entity boundaries
            if pred > 0:  # If it's an entity
                if pred % 2 == 1:  # If it's a B- label
                    if current_entity is not None:
                        entities.append({
                            "text": current_text.strip(),
                            "label": current_entity
                        })
                    current_entity = pred
                    current_text = word
                else:  # If it's an I- label
                    current_text += " " + word
            else:  # If it's O
                if current_entity is not None:
                    entities.append({
                        "text": current_text.strip(),
                        "label": current_entity
                    })
                current_entity = None
                current_text = ""
        
        # Add the last entity if there is one
        if current_entity is not None:
            entities.append({
                "text": current_text.strip(),
                "label": current_entity
            })
        
        return entities 