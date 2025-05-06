from django.urls import path
from . import views

urlpatterns = [
    # Define the URL pattern for the topic modeling page
    path('fetch_topics/', views.fetch_topics, name='topics'),
    path('get_papers/', views.get_recommendations, name = 'papers')# Topics page for topic modeling results
]