from django.conf.urls import url
from . import views
urlpatterns = [url(r'^$', views.index, name='index'),
               url(r'^contact/', views.contact, name='contact'),
               url(r'^string/', views.string, name='string'),
               url(r'^predict/', views.predict, name='predict'),
               url(r'^train/', views.train, name='train'),
               url(r'^predict2/', views.predict2, name='predict2'),
               url(r'^train2/', views.train2, name='train2'),
]
