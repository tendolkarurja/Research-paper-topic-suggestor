import os
import django
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "research_trends.settings") 
django.setup()

from scraper.models import ArxivPaper
import feedparser


api_query = "http://export.arxiv.org/api/query?search_query=cat:{}&sortBy=submittedDate&sortOrder=descending&max_results={}"
topics = ['cs.AI', 'cs.LG', 'quant-phy', 'cs.CL']

def fetch(category, max_results = 50):
    url = api_query.format(category, max_results)
    feed = feedparser.parse(url)
    papers = []
    
    for entry in feed.entries:
        title = entry.title.strip().replace('\n', ' ')
        abstract = entry.summary.strip().replace('\n', ' ')
        
        papers.append({'category':category, 'title':title, 'abstract':abstract})
    
    return papers

def store_in_db(papers):
    for paper in papers:
        if not ArxivPaper.objects.filter(title=paper['title'], abstract = paper['abstract']).exists():
            ArxivPaper.objects.create(category = paper['category'], title = paper['title'], abstract = paper['abstract'])

categories = ["cs.CV", "cs.LG", "cs.CL", "cs.AI", "cs.RO"]
for cat in categories:
    print(f"Fetching {cat}...")
    papers = fetch(cat, max_results=50)
    store_in_db(papers)