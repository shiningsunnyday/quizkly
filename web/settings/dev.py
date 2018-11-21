from .common import *


BROKER_URL = 'redis://'
CELERY_RESULT_BACKEND = 'redis://'

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '7sk#y*i2+kmzj&%%8!e)$d!q8-ybkm9i9o!at+pb_0p^lsr@dn'

ALLOWED_HOSTS = []

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

for tmpl in TEMPLATES:
    if 'OPTIONS' in tmpl:
        tmpl['OPTIONS']['debug'] = True
    else:
        tmpl['OPTIONS'] = {'debug':True}

# Database
# https://docs.djangoproject.com/en/1.7/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': join(BASE_DIR, 'db.sqlite3'),
    }
}
