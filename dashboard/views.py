

# Create your views here.
from django.shortcuts import render
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64

def trending_topics(request):
    # Dummy data for now
    topics = ['AI', 'ML', 'Quantum Computing', 'NLP']
    text = " ".join(topics)
    
    wordcloud = WordCloud(width=400, height=200).generate(text)
    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return render(request, 'dashboard/trending.html', {'wordcloud': img_str})
