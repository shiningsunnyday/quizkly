"""
Django settings for RevUP_Web project.

For more information on this file, see
https://docs.djangoproject.com/en/1.7/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.7/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
from os import getenv
from os.path import dirname, join


#Directories

PROJECT_ROOT = dirname(dirname(__file__))
BASE_DIR = os.getenv('BASE_DIR', os.path.join(PROJECT_ROOT,'local')) #dynamic data directory

MEDIA_URL = '/media/'
STATIC_URL = '/static/'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS':  [join(PROJECT_ROOT, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ]
        }
    }
]

STATIC_ROOT = os.path.join(BASE_DIR, 'static/')
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')



REVUP_MAX_UPLOAD = 42 * 1024 * 1024

# Authentication User Model
LOGIN_URL = 'user_login'

# Application definition
INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.admindocs',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'revup.common',
    'revup.web',
    'revup.rest'
)

MIDDLEWARE = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    #'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.security.SecurityMiddleware'
)

X_FRAME_OPTIONS= 'DENY'

ROOT_URLCONF = 'revup.web.urls'

WSGI_APPLICATION = 'revup.web.wsgi.application'

#REST
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    )
}

#Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format' : "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            'datefmt' : "%d/%b/%Y %H:%M:%S"
        },
    },
    'handlers': {
        'null': {
            'level':'DEBUG',
            'class':'logging.NullHandler',
        },
        'logfile': {
            'level':'DEBUG',
            'class':'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logfile"),
            'maxBytes': 50000,
            'backupCount': 2,
            'formatter': 'standard',
        },
        'console':{
            'level':'INFO',
            'class':'logging.StreamHandler',
            'formatter': 'standard'
        },
    },
    'loggers': {
        'django': {
            'handlers':['console'],
            'propagate': True,
            'level':'WARN',
        },
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}

# Internationalization
# https://docs.djangoproject.com/en/1.7/topics/i18n/

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True


#Generator
def _readEnvInt(key,default):
    try:
        return int(os.getenv(key,default))
    except ValueError:
        return default

_revupGenType = os.getenv("REVUP_GEN","core").lower()
REVUP_MKGEN = None

def _loadFakeGen():
    from revup.common.mock.fakegen import FakeGen
    _fakeGenSettings = {
        'DELAY': max(_readEnvInt("REVUP_FAKE_GEN_DELAY",0),0),
        'ERROR_PROB_RANGE': max(_readEnvInt("REVUP_FAKE_GEN_ERROR_PROB_RANGE",0),0)
    }
    return lambda: FakeGen(_fakeGenSettings)

def _loadCoreGen():
    try:
        from revup.core.question_generator import QuestionGenerator
        _path = os.getenv("REVUP_MODELS",os.path.join(BASE_DIR,"models"))
        return lambda: QuestionGenerator(_path)
    except ImportError:
        import traceback
        print("Error loading core generator:")
        traceback.print_exc()
        return None

if _revupGenType == "core":
    REVUP_MKGEN = _loadCoreGen()
elif _revupGenType == "fake":
    REVUP_MKGEN = _loadFakeGen()
else:
    raise ValueError("Unknown gen type {}".format(_revupGenType))

if REVUP_MKGEN == None:
    print("Unable to load core gen, using fakegen")
    REVUP_MKGEN = _loadFakeGen()


REVUP_CHUNK_SIZE = 4

CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT=['json']
CELERY_TIMEZONE = 'Asia/Singapore'
CELERY_ENABLE_UTC = True

if _revupGenType == "core":
    CELERYD_CONCURRENCY = 1
