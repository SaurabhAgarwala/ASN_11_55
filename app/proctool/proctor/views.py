from proctor.models import Violation
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from users.models import Account

def get_violation(request):
    users = Account.objects.all()[1:]
    # print(users[1].violations.fullscreen)
    context = {
        'users': users
    }
    return render(request, 'proctor/violations.html', context)
    

@login_required(login_url="/users/login/")
def test_view(request):
    user = request.user
    context = {'user':user}
    return render(request, 'proctor/test_page.html', context)


def report_violation(request,id,vio):
    user = Account.objects.get(pk=id)
    if vio == "multiperson":
        # print(user.violations.multiperson)
        user.violations.multiperson += 1
        # # print('Working**********')
        # print(user.violations.multiperson)
    elif vio == "fullscreen":
        user.violations.fullscreen += 1
    elif vio == "screenfocus":
        user.violations.screenfocus += 1
    elif vio == "audio":
        user.violations.audio += 1
    elif vio == "facedir":
        user.violations.facedir += 1
    elif vio == "facesim":
        user.violations.facesim += 1
    elif vio == "othergadgets":
        user.violations.othergadgets += 1
    else:
        print("Not Found")
    user.violations.save()
    return HttpResponse("Successful")
   
def submit(request):
    user = request.user
    logout(request)
    print(user.is_active)
    user.is_active = False
    print(user.is_active)
    user.save()
    return render(request, 'proctor/submit.html')





# def student_exam(request):
#     context = {}
#     return render(request, 'proctor/student_exam_page.html', context)
