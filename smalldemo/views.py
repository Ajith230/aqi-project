from django.contrib import auth
from django.contrib.auth.models import User
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect

# Create your views here.
import program
from program import prediction
from program.machinelearning1 import get_message


def view(reqst):
    return render(reqst, 'redirect.html')


def indx(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['pass']
        user = auth.authenticate(username=username, password=password)
        if user:
            auth.login(request, user)
            # return render(request, 'homepage.html')
            # return HttpResponse(user.username)
            request.session['username'] = user.first_name
            return redirect(homepage)
        else:
            # messages.info(request, 'Invalid Credentials')
            # return HttpResponse('Login unSuccesfull')
            return render(request, 'index.html', {'message': 'Invalid credentials', 'username': username})

    else:
        return render(request, 'index.html')


def regster(request):
    if request.method == 'POST':
        firstname = request.POST['name']
        mail = request.POST['username']

        password1 = request.POST['pass']

        if (not User.objects.filter(username=mail).exists()):

            user = User.objects.create_user(username=mail,email=mail, password=password1,first_name=firstname)
            user.save()
            request.session['username'] = user.first_name

            return redirect(homepage)

        else:
            # messages.info(request, 'Username already registered')
            return render(request, 'register2.html', {'message': 'Email already registered'})

    else:
        return render(request, 'register2.html')


def homepage(request):
    if request.user.is_authenticated:
        try:
            delete_session(request)
            #pass
        except :
            print("erorr")
        return render(request, 'homepage.html', {'username': request.session['username']})
    else:
        return redirect('/new')


def user_logout(request):
    if request.user.is_authenticated:
        auth.logout(request)
    return redirect('/new')


def register2(request):
    if request.method == "POST":
        name = (request.POST['name']).strip()
        mail = request.POST['email'].strip()

        password1 = request.POST['password']
        password2 = request.POST['re_password']
        if password1 == password2:
            if not User.objects.filter(username=mail).exists():
                user = User.objects.create_user(username=mail,first_name=name, password=password2, email=mail)
                user.save()
                request.session['username'] = name
                return redirect(homepage)
            else:
                return render(request, 'register2.html', {'message': 'Email already registered','name':name,'email':mail,'password':password1})
        else:
            return render(request,'register2.html',{'message': 'Two password should be same','name':name,'email':mail,'password':password1})



    else:
        return render(request, 'register2.html')

def data1(request):
    if request.user.is_authenticated:
        try:
            delete_session(request)
        except :
            print("erorr")
        if request.method == 'POST' :
            request.session['so2']=request.POST['so2']
            request.session['no2'] = request.POST['no2']
            request.session['rspm'] = request.POST['rspm']
            request.session['spm'] = request.POST['spm']
            request.session['value-algorithm']=request.POST['algorithm']
            return redirect(value)
        else:

            return  render(request,'data1.html',{'username': request.session['username']})
    else:
        return redirect('/new')

def data2(request):
    if request.user.is_authenticated:
        try:
            delete_session(request)
        except :
            print("erorr")
        if request.method == 'POST' :
            request.session['so2']=request.POST['so2']
            request.session['no2'] = request.POST['no2']
            request.session['rspm'] = request.POST['rspm']
            request.session['spm'] = request.POST['spm']
            request.session['range-algorithm']=request.POST['algorithm']
            return redirect(range)
        else:

            return  render(request,'data2.html',{'username': request.session['username']})
    else:
        return redirect('/new')



def value(request):

    if request.user.is_authenticated:
        if request.method == "POST":
            request.session['range-algorithm'] = request.POST['algorithm']

            return redirect(range)

        else:

            try:
                #value=0
                so2=float(request.session['so2'])
                no2=float(request.session['no2'])
                rspm=float(request.session['rspm'])
                spm=float(request.session['spm'])
                if request.session['value-algorithm'] == 'lr':
                    value=prediction.predict_linear_regression(so2=so2,no2=no2,rspm=rspm,spm=spm)
                    return render(request,'value.html',{'value':round(value,3),'username': request.session['username']})
                elif request.session['value-algorithm'] == 'svm':
                    value = prediction.predict_SVModel(so2=so2, no2=no2, rspm=rspm, spm=spm)
                    return render(request, 'value.html', {'value': round(value, 3),'username': request.session['username']})
                else:
                    value = prediction.predict_random_regressor(so2=so2, no2=no2, rspm=rspm, spm=spm)
                    return render(request, 'value.html', {'value': round(value, 3),'username': request.session['username']})



            except NameError as e1:
                print(e1)
                return redirect(data1)
                #pass
            except KeyError as e2:
                print(e2)
                #pass
                return redirect(data1)
    else:
        return redirect('/new')

def range(request):
    try:
        request.method
    except:
        request.method = "GET"
    if request.user.is_authenticated:
        if request.method=="POST":
            request.session['value-algorithm']=request.POST['algorithm']
            return redirect('/value')
        else:
            try:
                # value=0
                so2 = float(request.session['so2'])
                no2 = float(request.session['no2'])
                rspm = float(request.session['rspm'])
                spm = float(request.session['spm'])

                if request.session['range-algorithm'] == 'lr':
                    value = prediction.predict_logistic_regression(so2=so2, no2=no2, rspm=rspm, spm=spm)

                    return render(request, 'range.html', {'value': value,'username': request.session['username']})
                elif request.session['range-algorithm'] == 'rfc':
                    value = prediction.predict_random_classifier(so2=so2, no2=no2, rspm=rspm, spm=spm)
                    return render(request, 'range.html', {'value': value,'username': request.session['username']})
            except NameError:
                #pass
                return redirect(data2)
            except KeyError:
                pass
                return redirect(data2)
                #print(request.method)




    else:
        return redirect('/new')

def delete_session(request):
    del request.session['so2']
    del request.session['no2']
    del request.session['rspm']
    del request.session['spm']