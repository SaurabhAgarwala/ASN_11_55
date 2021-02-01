from django.shortcuts import render, redirect
from django.http import HttpResponse
from users.models import Account

def violation(request):
    users = Account.objects.all()[1:]
    print(users[1].violations.fullscreen)
    context = {
        'users': users
    }
    return render(request, 'proctor/view_all_violations.html', context)
    
def test_view(request):
    context = {}
    return render(request, 'proctor/test_page.html', context)

def student_exam(request):
    context = {}
    return render(request, 'proctor/student_exam_page.html', context)

def login(request):
    context = {}
    return render(request, 'proctor/login.html', context)

def logout(request):
    context = {}
    return render(request, 'proctor/logout.php', context)