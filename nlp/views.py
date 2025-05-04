from django.shortcuts import render

# Create your views here.

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
from .preprocessing import get_arxiv_papers, parse_arxiv_response

# Load models once globally
sentence_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
bert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

def lda_topic_modeling(papers):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(papers)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    topics = []
    for topic in lda.components_:
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])
    return topics

def bertopic_modeling(papers):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(papers)
    return topic_model.get_topic_info().to_dict()

def recommend_topics(query, papers):
    query_embedding = sentence_model.encode([query])
    paper_embeddings = sentence_model.encode(papers)
    similarities = cosine_similarity(query_embedding, paper_embeddings)
    most_similar_index = similarities.argmax()
    return papers[most_similar_index]

def scibert_similarity_search(texts, query):
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    query_input = bert_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        text_outputs = bert_model(**inputs).last_hidden_state.mean(dim=1)
        query_output = bert_model(**query_input).last_hidden_state.mean(dim=1)
    similarity = cosine_similarity(query_output.numpy(), text_outputs.numpy())
    most_similar_index = similarity.argmax()
    return texts[most_similar_index]

from django.shortcuts import render
from .preprocessing import get_arxiv_papers, parse_arxiv_response

# your existing model loading and functions remain here...

def fetch_topics(request):
    query = request.GET.get('query', 'Machine Learning')
    response_text = get_arxiv_papers(query)
    if not response_text:
        return JsonResponse({'error': 'Failed to fetch papers from arXiv'})

    papers = parse_arxiv_response(response_text)
    lda_topics = lda_topic_modeling(papers)
    bertopic_info_df = bertopic_modeling(papers)
    bertopic_topics = bertopic_info_df.to_dict(orient='records')
    recommended_topic = recommend_topics(query, papers)
    scibert_recommendation = scibert_similarity_search(papers, query)

    context = {
        'query': query,
        'lda_topics': lda_topics,
        'bertopic_topics': bertopic_topics,
        'recommended_topic': recommended_topic,
        'scibert_recommendation': scibert_recommendation
    }
    return render(request, 'papers/topics.html', context)
