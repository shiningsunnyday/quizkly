default_app_config = 'revup.common.apps.RevupCommonConfig'
#Run app init when django loads the common app
from .celery import app as celery_app
