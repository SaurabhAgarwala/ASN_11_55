from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import RegistrationForm, LoginForm
from .models import MyAccountManager, Account
from django.http import HttpResponse

def signup_view(request):
    context = {}
    if request.method == 'POST':
        registration_form = RegistrationForm(request.POST)
        if registration_form.is_valid():
            account = registration_form.save()
            user = Account.objects.get(email=account.email)
            user.save()
            login(request, account)
            # return redirect('users:userpage')
            return HttpResponse('Signed Up')
        else:
            context['form'] = registration_form
    else:
        form = RegistrationForm()
        context['form'] = form
    return render(request, 'users/signup_page.html', context)
    


def login_view(request):
    context = {}
    if request.user.is_authenticated:
        return redirect('application status')
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = MyAccountManager.normalize_email(form.cleaned_data['email'])
            password = request.POST['password']
            user = authenticate(email = email, password = password)
            if user:
                login(request, user)
                # return redirect('users:userpage')
                return HttpResponse('Logged In')
    else:
        form = LoginForm()
    context['form'] = form
    return render(request, 'users/login_page.html', context)

def logout_view(request):
        logout(request)
        return HttpResponse('Logged Out')
