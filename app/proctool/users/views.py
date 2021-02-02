from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import RegistrationForm, LoginForm
from .models import MyAccountManager, Account
from django.http import HttpResponse
from proctor.models import Violation

def signup_view(request):
    context = {}
    if request.method == 'POST':
        registration_form = RegistrationForm(request.POST, request.FILES)
        if registration_form.is_valid():
            account = registration_form.save()
            user = Account.objects.get(email=account.email)
            user.violations = Violation()
            user.save()
            login(request, account)
            return redirect('proctor:test')
            # return HttpResponse('Signed Up')
        else:
            context['form'] = registration_form
    else:
        form = RegistrationForm()
        context['form'] = form
    return render(request, 'users/home.html', context)
    


def login_view(request):
    context = {}
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = MyAccountManager.normalize_email(form.cleaned_data['email'])
            password = request.POST['password']
            user = authenticate(email = email, password = password)
            if user:
                login(request, user)
                return redirect('proctor:test')
                # return HttpResponse('Logged In')
    else:
        form = LoginForm()
    context['form'] = form
    return render(request, 'users/home.html', context)

def logout_view(request):
        logout(request)
        return redirect('users:login')
