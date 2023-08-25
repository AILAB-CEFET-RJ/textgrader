from django import forms
from .models import Essay, Theme


# creating a form
class EssaysForm(forms.ModelForm):
	# create meta class
	class Meta:
		# specify model to be used
		model = Essay

		# specify fields to be used
		fields = [
			"title",
			"content",
			"theme"
		]

		widgets = {
			"content": forms.Textarea(attrs={'rows':20, 'cols':10, 'class':'form-control', 'type':'text'}),
			"title": forms.TextInput(attrs={'class':'form-control', 'type':'text'})
		}


# creating a form
class EssaysCorrectForm(forms.ModelForm):

	# create meta class
	class Meta:
		# specify model to be used
		model = Essay

		# specify fields to be used
		fields = [
			"grade",
			"comments"
		]
		widgets = {
			"comments": forms.TextInput(attrs={'rows':20, 'cols':10, 'class':'form-control', 'type':'text'}),
			"grade": forms.TextInput(attrs={'class':'form-control', 'type':'text'})
		}



# creating a form
class ThemeForm(forms.ModelForm):

	# create meta class
	class Meta:
		# specify model to be used
		model = Theme

		# specify fields to be used
		fields = [
			"title",
			"context"
		]

		widgets = {
			"context": forms.Textarea(attrs={'rows':20, 'cols':10, 'class':'form-control', 'type':'text'}),
			"title": forms.TextInput(attrs={'class':'form-control', 'type':'text'})
		}
