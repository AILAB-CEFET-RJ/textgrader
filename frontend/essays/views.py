from django.http import HttpResponse
from django.shortcuts import render,redirect, get_object_or_404, HttpResponseRedirect, reverse
from .models import Essay, Theme
from django.http import Http404
from .forms import EssaysForm, ThemeForm, EssaysCorrectForm
from django.views.generic.list import ListView
from django.views.generic.edit import CreateView, UpdateView
from django.views.generic.detail import DetailView
from django import forms



# home view
class Home(ListView):
    model = Essay
    template_name = 'essays/base.html'

class ThemeList(ListView):
    model = Theme
    template_name = "themes/theme_list.html"

class ThemeCreate(CreateView):
    model = Theme
    template_name = "themes/theme_form.html"
    initial = {"key": "value"}
    form_class = ThemeForm

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {"form": form})

class ThemeDetail(DetailView):
    model = Theme
    fields = '__all__'
    template_name = "themes/theme_detail.html"

class EssayCreate(CreateView):
    model = Essay
    #fields = ['title','content']
    template_name = "essays/essay_form.html"
    initial = {"key": "value"}
    form_class = EssaysForm

    def get_success_url(self):
        return reverse('home')
    
    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {"form": form})

class EssayDetail(DetailView):
    model = Essay
    fields = '__all__'
    template_name = "essays/essay_detail.html"

    def get_success_url(self):
        return reverse('home')


class EssayList(ListView):
    model = Essay
    template_name = "essays/essay_list.html"
    fields = ['title', 'creation_date', 'theme']

    def get_success_url(self):
        return reverse('home')


class EssayCorrect(UpdateView):
    model = Essay
    template_name = "essays/essay_correct.html"
    form_class = EssaysCorrectForm

    def get_success_url(self):
        return reverse('home')