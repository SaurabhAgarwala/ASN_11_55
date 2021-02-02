from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from users.models import Account

def get_violation(request):
    users = Account.objects.all()[1:]
    print(users[1].violations.fullscreen)
    context = {
        'users': users
    }
    return render(request, 'proctor/violations.html', context)
    

@login_required(login_url="/users/login/")
def test_view(request):
    user = request.user
    context = {'user':user}
    return render(request, 'proctor/test_page.html', context)


# def report_violation(request):


# def student_exam(request):
#     context = {}
#     return render(request, 'proctor/student_exam_page.html', context)
