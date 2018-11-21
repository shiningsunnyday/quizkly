from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate, login, logout, get_user_model

from rest_framework import status, generics
from rest_framework.exceptions import ParseError, AuthenticationFailed
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from rest_framework.reverse import reverse
from rest_framework.metadata import SimpleMetadata
from rest_framework.permissions import IsAuthenticated, IsAdminUser

from ..common.models import Document
from .serializer import DocumentSerializer, DocumentOverviewSerializer, UserSerializer
from ..common.tasks import process_doc

@api_view(('GET',))
def api_root(request, format=None):
    return Response({
        'users': reverse('revupuser-list', request=request, format=format),
        'login': reverse('login', request=request, format=format),
        'logout': reverse('logout', request=request, format=format),
        'documents': reverse('document-list', request=request, format=format)
    })

class DocumentList(generics.ListCreateAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = DocumentOverviewSerializer
    def get_queryset(self):
        queryset = Document.objects.all()
        if not self.request.user.is_staff:
            queryset = queryset.filter(owner=self.request.user)
        return queryset

    def perform_create(self, serializer):
        doc = serializer.save(owner=self.request.user)
        process_doc(doc.id)

class DocumentDetail(generics.RetrieveDestroyAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer

class UserList(generics.ListAPIView):
    permission_classes = (IsAdminUser,)
    #permission_classes = (IsAuthenticated,)
    queryset = get_user_model().objects.all()
    serializer_class = UserSerializer


class UserDetail(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)
    queryset = get_user_model().objects.all()
    serializer_class = UserSerializer

class TestLogin(APIView):
    permission_classes = (IsAdminUser,)

    def get(self, request, format=None):
        content = {
            'user': unicode(request.user),  # `django.contrib.auth.User` instance.
            'auth': unicode(request.auth),  # None
        }
        return Response(content)

class Login(APIView):
    class Metadata(SimpleMetadata):
        actions = {
            'POST': {
                'username': {
                    'type': 'string',
                    'required': True,
                    'label': 'Username'
                },
                'password': {
                    'type': 'string',
                    'required': True,
                    'label': 'Password'
                }
            }
        }
        def determine_metadata(self,request,view):
            metadata = super(Login.Metadata,self).determine_metadata(request,view)
            metadata['actions'] = self.actions
            return metadata

    metadata_class = Metadata
    parser_classes = (JSONParser,)

    def post(self,request,format=None):
        if 'username' not in request.data:
            raise ParseError('Username not provided')
        username = request.data['username']

        if 'password' not in request.data:
            raise ParseError('Password not provided')
        password = request.data['password']
        
        user = authenticate(username=username,password=password)
        if user is None:
            raise AuthenticationFailed('Username/password invalid.')

        if not user.is_active:
            raise AuthenticationFailed('Account disabled.')

        login(request,user)
        sz = UserSerializer(user)
        return Response(sz.data)

class Logout(APIView):
    class Metadata(SimpleMetadata):
        actions = {
            'POST': {}
        }
        def determine_metadata(self,request,view):
            metadata = super(Login.Metadata,self).determine_metadata(request,view)
            metadata['actions'] = self.actions
            return metadata
    metadata_class = Metadata
    
    def post(self, request, format=None):
        logout(request)
        return Response()
