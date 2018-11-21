from django.conf.urls import url, include
from django.conf.urls.static import static
from django.conf import settings

from revup.web import views

urlpatterns = [
    url(r'^(?:index)?$', views.index, name='index'),
    url(r'^index/$', views.index, name='index'),
    url(r'^list/$', views.list, name='list'),
    url(r'^upload/$', views.upload,  name='upload'),
    url(r'^login/$', views.user_login, name='user_login'),
    url(r'^signup/$', views.signup, name='signup'),
    url(r'^logout/$', views.user_logout, name='user_logout'),
    url(r'^document/(\d+)/$', views.document, name='document'),
    url(r'^docjson/(\d+)/$', views.docjson, name='docjson'),
    url(r'^api/', include('revup.rest.urls')),
#    url(r'^api-auth/', include('rest_framework.urls'))
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) \
  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

