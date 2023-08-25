from django.db import models
import datetime
from django.utils import timezone

class Theme(models.Model):
    title = models.CharField(max_length = 200, verbose_name="Título")
    context = models.TextField(verbose_name="Contexto")
    creation_date = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return "/essays/themes/%i/" % self.id


class Essay(models.Model):
    UOL = "UOL"
    BRASIL_ESCOLA = "BRASIL_ESCOLA"
    ORIGIN_CHOICES = [
        (UOL, "UOL"),
        (BRASIL_ESCOLA, "Brasil Escola")
    ]

    title = models.CharField(max_length = 200, verbose_name="Título")
    content = models.TextField(default="", verbose_name="Conteúdo")
    grade = models.FloatField(default=0.0, verbose_name="Nota")
    origin = models.CharField(max_length=20, choices=ORIGIN_CHOICES, default=UOL)
    comments = models.TextField(default="", verbose_name="Correção")
    creation_date = models.DateTimeField(auto_now_add = True, verbose_name="Data de criação")
    theme = models.ForeignKey(Theme, on_delete=models.CASCADE, verbose_name="Tema")
    corrected = models.BooleanField(default=False, verbose_name="Corrigido")

    def __str__(self):
        return self.title
    
    def was_published_recently(self):
        return self.creation_date >= timezone.now() - datetime.timedelta(days=1)
