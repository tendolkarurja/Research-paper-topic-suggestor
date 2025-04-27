import requests
from bs4 import BeautifulSoup

def scrape_arxiv(query="machine learning"):
    url = f"https://arxiv.org/search/?query={query}&searchtype=all"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    papers = []
    for result in soup.select(".arxiv-result"):
        title = result.h1.text.strip()
        abstract = result.find("span", class_="abstract-full").text.strip()
        papers.append({'title': title, 'abstract': abstract})
    return papers