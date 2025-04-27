from celery import shared_task
from .scrapper import scrape_arxiv

@shared_task
def scheduled_scrape():
    papers = scrape_arxiv()
    # Save to DB (you'll make a Paper model)
    for paper in papers:
        # Save logic here
        pass