from django.urls import path
from . import views

urlpatterns = [
    # Define the URL pattern for the topic modeling page
    path('fetch_topics/', views.fetch_topics, name='topics'),  # Topics page for topic modeling results
]