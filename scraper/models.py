from django.db import models


class ArxivPaper(models.Model):
    title = models.TextField()
    abstract = models.TextField()
    category = models.CharField(max_length=20)

    def __str__(self):
        return self.title
