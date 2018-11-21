import logging

log = logging.getLogger(__name__)

# Create your views here.
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponse
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.auth.decorators import login_required
from django.urls import reverse

from revup.common.models import Document, RevupUserInfo

from revup.web.forms import DocumentForm

from ..common.tasks import process_doc


@require_http_methods(['GET','POST'])
def index(request):
    if request.user.is_authenticated:
        return redirect('list')

    else:
        return render(request, 'revup/index.html')

@require_http_methods(['GET','POST'])
def signup(request):
    if request.user.is_authenticated:
        return redirect('list')
    
    if request.method == 'GET':
        return render(request, 'revup/signup.html')

    # TODO: Check for failure, e.g. user existing, etc
    email = request.POST['email']
    password = request.POST['password']
    first_name = request.POST['first_name']
    last_name = request.POST['last_name']
    school = request.POST['school']
    user = get_user_model().objects.create_user(
        email, email, password,
        first_name=first_name, last_name=last_name)
    userInfo = RevupUserInfo(user=user, school=school)
    userInfo.save()
    return redirect('user_login')

@require_http_methods(['GET','POST'])
def user_login(request):
    nexturl = request.POST.get('next') or request.GET.get('next')

    if request.user.is_authenticated:
        return redirect(nexturl or 'list')
        
    if request.method == 'GET':
        return render(request, 'revup/login.html', {'next': nexturl })

    email = request.POST['email']
    password = request.POST['password']
    user = authenticate(username=email, password=password)

    if user is None:
        return render(request, 'revup/login.html',
            {'login_error': 'Wrong username/password', 'next': nexturl})
    
    if not user.is_active:
        return render(request, 'revup/login.html',
            {'login_error': 'Account disabled.', 'next': nexturl})

    login(request, user)
    return redirect(nexturl or 'list')

def user_logout(request):
    logout(request)
    return redirect('index')

@login_required
def upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.makeDoc()
            doc.status = Document.IN_QUEUE
            doc.owner = request.user
            doc.save()
            process_doc(doc.id)
            return redirect(doc)
    else:
        form = DocumentForm()
    return render(request, 'revup/upload.html', {'form':form})

@login_required
def list(request):
    # Load documents for the list page
    documents = Document.objects.filter(owner=request.user)
    # Render list page with the documents and the form
    return render(request,'revup/list.html',{'documents': documents})

@login_required
def document_old(request, docId):
    doc = get_object_or_404(Document, id=docId)
    return render(request, 'revup/questionscar.html', {'doc': doc})

#@login_required
def document(request, docId):
    doc = get_object_or_404(Document, id=docId)
    return render(request, 'revup/questions.html')

@login_required
def docjson(request, docId):
    try:
        startId = int(request.GET.get('start', 0))
    except ValueError:
        startId = 0

    try:
        endId = int(request.GET.get('end', -1))
    except ValueError:
        endId = -1

    doc = get_object_or_404(Document, id=docId)
    qns = doc.questions.all()
    jsonQns = []
    if startId > 0:
        qns = qns.filter(qnNo__gte=startId)
    if endId >= 0:
        qns = qns.filter(qnNo__lte=endId)

    for qn in qns:
        distractors = []
        for d in qn.distractors.all():
            distractors.append(d.distractor)
        jsonQns.append({
            'qnText': qn.qnText,
            'ans': qn.ans,
            'distractors': distractors,
            'id': qn.qnNo
        })
    jsonDoc = {
        'qns': jsonQns,
        'status': doc.status
    }
    return JsonResponse(jsonDoc, safe=False)
