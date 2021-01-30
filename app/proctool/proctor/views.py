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
    