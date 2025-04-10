# LegalNERModel

A Named Entity Recognition (NER) model specifically trained for legal documents and texts.

## Overview

LegalNERModel is a specialized natural language processing tool designed to identify and classify named entities in legal documents. It recognizes entities such as persons, organizations, legislation references, court cases, dates, locations, and legal-specific terminology that standard NER models might miss.

## Features

- Specialized entity recognition for legal documents
- Pre-trained on diverse legal corpora including contracts, court opinions, legislation, and legal briefs
- Support for multiple legal jurisdictions and formats
- Higher accuracy for legal entity recognition compared to general-purpose NER models
- Easy integration with popular NLP pipelines

## Installation

```bash
pip install legal-ner-model
```

Or clone the repository:

```bash
git clone https://github.com/NikolasRoufas/LegalNERModel.git
cd LegalNERModel
pip install -e .
```

## Quick Start

```python
from legal_ner_model import LegalNER

# Initialize the model
legal_ner = LegalNER()

# Process a legal text
text = "The plaintiff, John Doe, filed a motion on January 15, 2023, citing Smith v. Jones (2018) and Section 230 of the Communications Decency Act."
entities = legal_ner.extract_entities(text)

# Display recognized entities
for entity in entities:
    print(f"Entity: {entity.text}, Type: {entity.label_}, Position: {entity.start_char}-{entity.end_char}")
```

## Entity Types

The model recognizes the following entity types:

- `PERSON`: Individual names
- `ORG`: Organizations, companies, institutions
- `LAW`: Laws, acts, statutes, regulations
- `CASE`: Case citations and references
- `DATE`: Dates and time expressions
- `LOC`: Locations, jurisdictions
- `COURT`: Courts and tribunals
- `ROLE`: Legal roles (judge, plaintiff, defendant, etc.)
- `PROVISION`: Sections, articles, and clauses of legal documents

## Training Data

The model was trained on a diverse corpus of legal texts including:

- Court opinions from multiple jurisdictions
- Legislation and statutory materials
- Contracts and legal agreements
- Legal briefs and memoranda
- Academic legal articles

## Model Performance

| Entity Type | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| PERSON     | 0.94      | 0.92   | 0.93     |
| ORG        | 0.89      | 0.85   | 0.87     |
| LAW        | 0.92      | 0.88   | 0.90     |
| CASE       | 0.95      | 0.93   | 0.94     |
| DATE       | 0.96      | 0.97   | 0.96     |
| LOC        | 0.91      | 0.89   | 0.90     |
| COURT      | 0.93      | 0.91   | 0.92     |
| ROLE       | 0.88      | 0.84   | 0.86     |
| PROVISION  | 0.90      | 0.87   | 0.88     |
| **Overall**| **0.92**  | **0.90**| **0.91** |

## Advanced Usage

### Custom Entity Types

```python
from legal_ner_model import LegalNER, EntityType

# Define custom entity types
custom_types = [
    EntityType("JUDGE", "Names of judges"),
    EntityType("PLAINTIFF", "Names of plaintiffs"),
    EntityType("DEFENDANT", "Names of defendants")
]

# Initialize with custom entity types
legal_ner = LegalNER(custom_entity_types=custom_types)

# Train with custom types
legal_ner.train("path/to/training/data")
```

### Integration with spaCy

```python
import spacy
from legal_ner_model import legal_ner_pipe

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Add legal NER component
nlp.add_pipe("legal_ner", after="ner")

# Process text
doc = nlp("The Court in Brown v. Board of Education (1954) overturned Plessy v. Ferguson (1896).")

# Display entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
```

## Citing

If you use LegalNERModel in your research, please cite:

```
@software{roufas2025legalner,
  author = {Roufas, Nikolaos},
  title = {LegalNERModel: Named Entity Recognition for Legal Documents},
  year = {2025},
  url = {https://github.com/NikolasRoufas/LegalNERModel}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Nikolas Roufas - [GitHub](https://github.com/NikolasRoufas)

Project Link: [https://github.com/NikolasRoufas/LegalNERModel](https://github.com/NikolasRoufas/LegalNERModel)
