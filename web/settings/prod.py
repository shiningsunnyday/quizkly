from .common import *


BROKER_URL = 'amqp://compute:rabbit+compute@10.117.242.221:5672/compute_vhost'
CELERY_RESULT_BACKEND = 'redis://:redis&revup64+@10.117.242.221:6379/0'


# SECURITY WARNING: keep the secret key used in production secret!
# TODO: Replace with reading from env
SECRET_KEY = '98adsj)uda0-9d8jca-8d-aca-s0ck0-i8aj1laspdp00cascooa'

ALLOWED_HOSTS = [ '.revup.cc', '127.0.0.1' ]

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

TEMPLATE_DEBUG = False

# Database
# https://docs.djangoproject.com/en/1.7/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'revupdb',
        'USER': 'revup',
        'PASSWORD': 'psql@revup',
        'HOST': '' #'10.117.242.221',
    }
}
SESSION_COOKIE_SECURE = True
