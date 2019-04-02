from django.db import models

# Create your models here.


class userInfo(models.Model):
    def __str__(self):
        return self.name

    name = models.CharField(max_length=64,unique=True, primary_key=True)
    pwd = models.CharField(max_length=64)
    email = models.EmailField(unique=True)
    type = models.CharField(max_length=64, default='普通用户')


class managerInfo(models.Model):
    def __str__(self):
        return self.id

    id = models.CharField(max_length=64, unique=True, primary_key=True)
    pwd = models.CharField(max_length=64)
    type = models.CharField(max_length=64)




