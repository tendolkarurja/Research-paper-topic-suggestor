from django.shortcuts import render

# Create your views here.
def show_recommendations(request):
    papers = [
        {'title': 'AI in Healthcare', 'authors': 'John Doe, Jane Smith', 'link': 'https://example.com/paper1'},
        {'title': 'Machine Learning for Climate Change', 'authors': 'Alice Johnson', 'link': 'https://example.com/paper2'},
        {'title': 'Quantum Computing and AI', 'authors': 'Bob Lee', 'link': 'https://example.com/paper3'}
    ]
    return render(request, 'recommender/recommend.html', {'papers':papers})