from __future__ import absolute_import
from celery import Celery
from celery.signals import task_prerun, task_postrun
from django.conf import settings

app = Celery('worker')
app.config_from_object('django.conf:settings')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)