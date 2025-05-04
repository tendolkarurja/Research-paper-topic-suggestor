import django
import os
import sys
from django.conf import settings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the Python path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "research_trends.settings")
django.setup()


import pandas as pd
import spacy
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

from scraper.models import ArxivPaper  # Replace with your app name

# Load resources
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
NER = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')


# Load DB into DataFrame
def load_papers_from_db():
    papers = ArxivPaper.objects.all().values('title', 'category', 'abstract')
    df = pd.DataFrame(papers)
    df = df.dropna()
    df['title'] = df['title'].str.lower()
    df['category'] = df['category'].str.lower()
    df['abstract'] = df['abstract'].str.lower()
    return df


# Preprocess text
def lemmatize_and_extract_entities(text):
    doc = NER(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    entities = [ent.text.lower() for ent in doc.ents]
    return lemmatized_text, entities


# Preprocess the DataFrame
def preprocess_dataframe(df):
    descriptions = []
    all_entities = []

    for _, row in df.iterrows():
        combined_text = row['abstract'] + " " + row['category']
        lemmatized_text, entities = lemmatize_and_extract_entities(combined_text)
        descriptions.append(lemmatized_text)
        all_entities.append(entities)

    df['lemmatized_description'] = descriptions
    df['entities'] = all_entities

    desc_emb = model.encode(descriptions, convert_to_tensor=False)
    title_emb = model.encode(df['title'].tolist(), convert_to_tensor=False)

    return {
        'df': df,
        'desc_emb': desc_emb,
        'paper_emb': title_emb,
        'entities': all_entities
    }

def relevant_papers(user_entry, df, desc_emb, paper_emb, entities, domain=None):
    user_input_lemmatized, user_input_entities = lemmatize_and_extract_entities(user_entry.lower())
    user_emb = model.encode([user_input_lemmatized], convert_to_tensor=False)
    similarity = util.cos_sim(user_emb, desc_emb)[0].cpu().numpy()

    df['similarity_score'] = similarity

    entity_similarity_scores = []
    for paper_entities in entities:
        entity_similarity = len(set(user_input_entities).intersection(set(paper_entities)))
        entity_similarity_scores.append(entity_similarity)

    df['entity_similarity_score'] = entity_similarity_scores

    if domain:
        domain_lem, domain_entities = lemmatize_and_extract_entities(domain.lower())
        domain_vec = model.encode(domain_lem, convert_to_tensor=True)
        title_vecs = model.encode(df['title'].tolist(), convert_to_tensor=True)
        domain_similarity = util.cos_sim(domain_vec, title_vecs)[0].cpu().numpy()
        df['domain_similarity'] = domain_similarity
        df = df[df['domain_similarity'] > 0.3]

    df['final_score'] = df['similarity_score'] + df['entity_similarity_score']
    return df.sort_values('final_score', ascending=False)

# if __name__ == "__main__":
#     # Load papers from database
#     df = load_papers_from_db()
    
#     print("Total records:", len(df))
#     print("Sample data:")
#     print(df.head())

import requests
import spacy
from xml.etree import ElementTree

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')

def get_arxiv_papers(query='machine learning', max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(base_url + search_query)
    return response.text if response.status_code == 200 else None

def clean_and_tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def parse_arxiv_response(response_text):
    root = ElementTree.fromstring(response_text)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
        paper = f"{title}. {abstract}"
        papers.append(clean_and_tokenize(paper))
    return papers

