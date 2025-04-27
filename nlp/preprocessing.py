import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_keywords(texts):
    tfidf = TfidfVectorizer(max_features=10)
    tfidf.fit(texts)
    return tfidf.get_feature_names_out()