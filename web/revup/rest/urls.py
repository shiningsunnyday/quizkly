from django.conf.urls import url, include
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    url(r'^$', views.api_root),
    url(r'^login/$',views.Login.as_view(),name='login'),
    url(r'^logout/$',views.Logout.as_view(),name='logout'),
    url(r'^testlogin/$', views.TestLogin.as_view(), name='test-login'),
    url(r'^users/$', views.UserList.as_view(), name='revupuser-list'),
    url(r'^users/(?P<pk>\d+)$', views.UserDetail.as_view(), name='revupuser-detail'),
    url(r'^documents/$', views.DocumentList.as_view(), name='document-list'),
    url(r'^documents/(?P<pk>\d+)$', views.DocumentDetail.as_view(), name='document-detail')
]

urlpatterns = format_suffix_patterns(urlpatterns)
