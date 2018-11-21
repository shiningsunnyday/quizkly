from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from rest_framework import serializers
from ..common.models import Document, Question

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = settings.AUTH_USER_MODEL
        fields = ('id','username','first_name','last_name')

class QuestionSerializer(serializers.ModelSerializer):
    distractors = serializers.StringRelatedField(many=True, read_only=True)
    class Meta:
        model = Question
        fields = ('doc', 'qnNo', 'qnText', 'ans', 'distractors')

class DocumentSerializer(serializers.ModelSerializer):
    questions = serializers.SerializerMethodField() #QuestionSerializer(many=True, read_only=True)
    class Meta:
        model = Document
        fields = ('id', 'status', 'questions')
    
    def get_questions(self,doc):
        data = self.context['request'].query_params
        qns = doc.questions.all()
        if 'start' in data:
            try:
                startId = int(data['start'])
                qns = qns.filter(qnNo__gte=startId)
            except ValueError:
                raise serializers.ValidationError('start is not an integer')

        if 'end' in data:
            try:
                endId = int(data['end'])
                qns = qns.filter(qnNo__lte=endId)
            except ValueError:
                raise serializers.ValidationError('end is not an integer')

        return QuestionSerializer(qns,many=True).data

class DocumentOverviewSerializer(serializers.HyperlinkedModelSerializer):
    detail = serializers.HyperlinkedIdentityField(view_name='document-detail')

    docmethod = serializers.ChoiceField(['file','text'], write_only=True)
    doctext = serializers.CharField(write_only=True, required=False)
    class Meta:
        model = Document
        fields = (
            'id', 'status', 'owner', 'detail',
            'docmethod', 'docfile', 'doctext'
        )
        read_only_fields = ('status', 'owner')
        extra_kwargs = {
            'docfile': {
                'write_only': True,
                'required': False,
                'max_length': settings.REVUP_MAX_UPLOAD
            },
        }
   
    def validate(self,data):
        if data['docmethod'] == 'text':
            if not 'doctext' in data:
                raise serializers.ValidationError('No text supplied!')
            fpath = default_storage.get_available_name('pasteUpload.txt')
            savedDoc = default_storage.save(fpath, ContentFile(data['doctext']))
            docfile = savedDoc
        else:
            if not 'docfile' in data:
                raise serializers.ValidationError('No file supplied!')
            docfile = data['docfile']
        
        return {
            'docfile': docfile,
            'status': Document.IN_QUEUE
        }
