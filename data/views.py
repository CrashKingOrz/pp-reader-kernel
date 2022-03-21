from django.http import HttpResponse
from .models import User
from .models import Query
from .models import Time
import json

def InsertUser(request):
    user = User()
    user.open_id = request.GET.get('open_id')
    user.name = request.GET.get('name')
    user.gender = request.GET.get('gender')
    user.phone_num = request.GET.get('phone_num')
    user.save()
    json_data = {'status':200,'id':user.id,'type':'user'}
    return HttpResponse(json.dumps(json_data, ensure_ascii=False))

def InsertQuery(request):
    query = Query()
    query.open_id = request.GET.get('open_id')
    query.Type = request.GET.get('type')
    query.content = request.GET.get('content')
    query.time = request.GET.get('time')
    query.result = request.GET.get('result')
    query.save()
    json_data = {'status':200,'id':query.id,'type':'query'}
    return HttpResponse(json.dumps(json_data, ensure_ascii=False))

def InsertTime(request):
    time = Time()
    time.open_id = request.GET.get('open_id')
    time.start_time = request.GET.get('start_time')
    time.end_time = request.GET.get('end_time')
    time.duration = request.GET.get('duration')
    time.save()
    json_data = {'status':200,'id':time.id,'type':'time'}
    return HttpResponse(json.dumps(json_data, ensure_ascii=False))

def getAllUserInfo(request):
    # open_id = request.GET.get('open_id')

    results = User.objects.filter()
    arr = []
    for res in results:
        content = {'id':res.id, 'open_id':res.open_id, 'name':res.name, 'gender':res.gender, 'phone_num':res.phone_num}
        arr.append(content)
    json_data = {'status':200, 'res':arr}
    # return HttpResponse('NB')
    return HttpResponse(json.dumps(json_data, ensure_ascii=False))
