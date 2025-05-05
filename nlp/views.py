from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from bertopic import BERTopic
import numpy as np
import torch

# Import preprocessing functions
from .preprocessing import get_arxiv_papers, parse_arxiv_response, load_papers_from_db, relevant_papers

# Load models once globally
sentence_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
bert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

def lda_topic_modeling(papers):
    """
    Perform LDA topic modeling on the provided list of papers.
    Each paper should be a string.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(papers)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    topics = []
    for topic in lda.components_:
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])  # Get top 5 terms
    return topics

def bertopic_modeling(papers):
    """
    Perform BERTopic modeling on the provided list of papers.
    Each paper should be a string.
    """
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(papers)
    topic_info = topic_model.get_topic_info()
    return topic_info.to_dict(orient='records')  # Return topic info as a list of dictionaries

def recommend_topics(query, papers):
    """
    Recommend the most similar paper to the query using cosine similarity.
    """
    query_embedding = sentence_model.encode([query])
    paper_embeddings = sentence_model.encode(papers)
    similarities = cosine_similarity(query_embedding, paper_embeddings)
    most_similar_index = similarities.argmax()
    return papers[most_similar_index]

def scibert_similarity_search(texts, query):
    """
    Use SciBERT model to find the most similar paper to the query based on cosine similarity.
    """
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    query_input = bert_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        text_outputs = bert_model(**inputs).last_hidden_state.mean(dim=1)
        query_output = bert_model(**query_input).last_hidden_state.mean(dim=1)
    similarity = cosine_similarity(query_output.numpy(), text_outputs.numpy())
    most_similar_index = similarity.argmax()
    return texts[most_similar_index]

def fetch_topics(request):
    """
    Fetch topics based on a query and return a response with topic information.
    """
    query = request.GET.get('query', 'Machine Learning')
    response_text = get_arxiv_papers(query)
    if not response_text:
        return JsonResponse({'error': 'Failed to fetch papers from arXiv'})

    papers = parse_arxiv_response(response_text)
    if not papers:
        return JsonResponse({'error': 'Failed to parse papers from arXiv response'})

    # LDA Topic Modeling
    lda_topics = lda_topic_modeling(papers)

    # BERTopic Modeling
    bertopic_info = bertopic_modeling(papers)
    
    # Recommend based on cosine similarity
    recommended_topic = recommend_topics(query, papers)
    
    # SciBERT similarity search
    scibert_recommendation = scibert_similarity_search(papers, query)

    # Return results to the frontend (render a template)
    context = {
        'query': query,
        'lda_topics': lda_topics,
        'bertopic_topics': bertopic_info,
        'recommended_topic': recommended_topic,
        'scibert_recommendation': scibert_recommendation
    }
    
    return render(request, 'nlp/topics.html', context)

from django.shortcuts import render
from nlp.preprocessing import load_papers_from_db, preprocess_dataframe, relevant_papers

df = load_papers_from_db()
processed = preprocess_dataframe(df)

def get_recommendations(request):
    user_query = request.GET.get('query', '')
    domain = request.GET.get('domain', None)

    suggested_topics = []
    if user_query:
        results = relevant_papers(
            user_query,
            processed['df'].copy(),
            processed['desc_emb'],
            processed['paper_emb'],
            processed['entities'],
            domain=domain
        )
        suggested_topics = results['title'].tolist()

    return render(request, 'recommender/recommend.html', {
        'suggested_topics': suggested_topics,
        'user_query': user_query
    })