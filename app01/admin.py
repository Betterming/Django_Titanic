from django.contrib import admin

# Register your models here.
from app01 import models


class Myadmin(admin.ModelAdmin):
    list_display = ('name', 'pwd')


admin.site.register(models.userInfo, Myadmin)

