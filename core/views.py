from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages

# core/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login
from .forms import CustomUserCreationForm  # Import your custom form

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # Save the user to the database
            login(request, user)  # Log the user in
            return redirect('/')
    else:
        form = CustomUserCreationForm()  # Use the custom form here
    return render(request, 'core/register.html', {'form': form})

def log_in(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data = request.POST)
        if form.is_valid():
            user = form.get_user()
            print("Authenticated")
            login(request, user)
            messages.success(request, 'Login Successful')
            return redirect('/dashboard')
        else:
            print("Error")
            messages.error(request, 'An error occured')
    else:
        form = AuthenticationForm()
    return render(request, 'core/login.html', {'form':form})
                


def home(request):
    return render(request, 'core/home.html')