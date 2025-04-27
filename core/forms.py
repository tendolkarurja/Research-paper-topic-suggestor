from django import forms
from django.contrib.auth import get_user_model

User = get_user_model()  # Get the custom user model

class CustomUserCreationForm(forms.ModelForm):
    class Meta:
        model = User  # Use the custom user model
        fields = ['username', 'password', 'email']  # Include any additional fields you may have added
        widgets = {
            'password': forms.PasswordInput(),
        }