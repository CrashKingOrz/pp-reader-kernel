from django.db import models

# Create your models here.

class User(models.Model):
    id = models.AutoField(primary_key=True)
    open_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=100)
    phone_num = models.CharField(max_length=100)
    class Meta:
        db_table = 'user'

class Query(models.Model):
    id = models.AutoField(primary_key=True)
    open_id = models.CharField(max_length=100)
    Type = models.CharField(max_length=100)
    content = models.CharField(max_length=100)
    time = models.DateTimeField(auto_now_add=True)
    result = models.CharField(max_length=100)
    class Meta:
        db_table = 'query'

class Time(models.Model):
    id = models.AutoField(primary_key=True)
    open_id = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(default=0)
    class Meta:
        db_table = 'time'