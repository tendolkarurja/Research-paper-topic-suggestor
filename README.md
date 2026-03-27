# Research-paper-topic-suggestor

# 📚 Research Paper Topic Suggestor

A Django-based NLP system that recommends relevant research papers using semantic similarity and entity-based matching.

---

## 🚀 Features

- Suggests research papers based on user input
- Uses Sentence Transformers for semantic similarity
- Applies NLP preprocessing (lemmatization, stopword removal)
- Enhances ranking using Named Entity Recognition (NER)
- Supports optional domain-based filtering
- Fetches papers from arXiv API

---

## 🏗️ Tech Stack

- Python, Django  
- spaCy, NLTK  
- Sentence Transformers  
- pandas  
- arXiv API  

---

## ⚙️ Setup

```bash
git clone https://github.com/tendolkarurja/Research-paper-topic-suggestor.git
cd Research-paper-topic-suggestor

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python manage.py migrate
python manage.py runserver
