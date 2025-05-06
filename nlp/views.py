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
from .preprocessing import get_arxiv_papers, parse_arxiv_response

# Load models once globally
sentence_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
bert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

def lda_topic_modeling(papers, num_topics=3, num_terms=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(papers)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics = []
    for topic in lda.components_:
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_terms:]])
    return topics

def bertopic_modeling(papers, num_topics=3):
    topic_model = BERTopic(top_n_words=num_topics)
    topics, _ = topic_model.fit_transform(papers)
    topic_info = topic_model.get_topic_info()
    return topic_info.to_dict(orient='records')


def recommend_topics(query, papers):
    """
    Recommend the most similar paper to the query using cosine similarity.
    """
    query_embedding = sentence_model.encode([query])
    paper_embeddings = sentence_model.encode(papers)
    similarities = cosine_similarity(query_embedding, paper_embeddings)
    most_similar_index = similarities.argmax()
    return papers[most_similar_index]

def scibert_similarity_search(texts, query, num_topics=3):
    """
    Use SciBERT model to find the top N most similar papers to the query based on cosine similarity.
    Returns a list of recommended texts.
    """
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    query_input = bert_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        text_outputs = bert_model(**inputs).last_hidden_state.mean(dim=1)
        query_output = bert_model(**query_input).last_hidden_state.mean(dim=1)
    
    similarity = cosine_similarity(query_output.numpy(), text_outputs.numpy())[0]
    
    # Get indices of top N similar texts
    top_indices = similarity.argsort()[-num_topics:][::-1]

    # Return those top N texts as a list
    recommended_texts = [texts[i] for i in top_indices]

    return recommended_texts


def fetch_topics(request):
    query = request.GET.get('query', 'Machine Learning')
    num_topics = int(request.GET.get('num_topics', 3))
    num_terms = int(request.GET.get('num_terms', 5))
    model_choice = request.GET.get('model_choice', 'both')
    

    response_text = get_arxiv_papers(query)
    if not response_text:
        return JsonResponse({'error': 'Failed to fetch papers from arXiv'})

    papers = parse_arxiv_response(response_text)
    if not papers:
        return JsonResponse({'error': 'Failed to parse papers from arXiv response'})

    context = {
        'query': query,
        'model_choice': model_choice,
    }

    if model_choice in ['lda', 'both']:
        lda_topics = lda_topic_modeling(papers, num_topics, num_terms)
        context['lda_topics'] = lda_topics

    if model_choice in ['bertopic', 'both']:
        bertopic_info = bertopic_modeling(papers, num_topics)
        context['bertopic_topics'] = bertopic_info

    # Recommendation and similarity search
    recommended_topic = recommend_topics(query, papers)
    scibert_recommendation = scibert_similarity_search(papers, query, num_topics)
    context.update({
        'recommended_topic': recommended_topic,
        'scibert_recommendation': scibert_recommendation
    })

    return render(request, 'nlp/topics.html', context)

