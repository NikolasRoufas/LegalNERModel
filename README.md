# Legal NER Model

A Named Entity Recognition (NER) model for legal texts using Transformer-based architecture in TensorFlow.

## Overview

This project implements a NER model specifically designed for legal texts using:
- BERT-based architecture (Legal-BERT)
- TensorFlow and Hugging Face's transformers library
- Custom loss function for handling class imbalance
- Support for IOB format entity labels

## Features

- Legal text preprocessing and tokenization
- Custom loss function with class weights
- Support for multiple entity types
- Early stopping and learning rate reduction
- Entity boundary detection and extraction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-ner.git
cd legal-ner
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in CSV format with columns 'text' and 'labels'
2. Run the training script:
```bash
python main.py
```

3. Use the model for prediction:
```python
from legal_ner_model import LegalNERModel

model = LegalNERModel(num_labels=9)  # Adjust based on your number of entity types
entities = model.predict("Your legal text here")
```

## Project Structure

```
legal-ner/
├── main.py                 # Main training script
├── legal_ner_model.py      # Model implementation
├── data_preprocessing.py   # Data preprocessing utilities
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Legal-BERT model from Hugging Face
- TensorFlow and Hugging Face's transformers library 