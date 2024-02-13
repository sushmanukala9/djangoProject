from django import forms

class HomeForm(forms.Form):
    your_csv_file_path = forms.CharField()
class UserlistForm(forms.Form):
    users = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple,label="Notify and subscribe users to this post:")
