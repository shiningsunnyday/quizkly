from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms import Form, FileInput, TextInput, Textarea, HiddenInput
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from revup.common.models import Document

class DocumentForm(Form):
    docFile = forms.FileField(
        widget=FileInput(attrs={'data-show-preview': 'false', 'class': 'file'}),
        label='Select a file.',
        help_text='max. 42 megabytes',
        required=False)
    docText = forms.CharField(
        widget=Textarea(attrs={'class': 'form-control'}),
        required=False)
    docMethod = forms.ChoiceField(
        widget=HiddenInput(attrs={'id': 'submissionType'}),
        choices=(
            ('pastePanel', 'Pasted'),
            ('filePanel', 'Uploaded')
        ),
        initial='filePanel')
    docTitle = forms.CharField(
            widget=TextInput(attrs={'class': 'form-control'}),
            required=False)

    def clean(self):
        cleaned_data = super(DocumentForm, self).clean()
        method = cleaned_data.get('docMethod')
        if method == 'filePanel':
            upFile = cleaned_data.get('docFile')
            if not upFile:
                raise ValidationError('Empty file input.')
            if upFile.size > MAX_SIZE:
                raise ValidationError('File too big.')
        else:
            upText = cleaned_data.get('docText')
            if not upText:
                raise ValidationError('Empty pasted input.')
            if len(upText.encode('utf-8')) > settings.REVUP_MAX_UPLOAD:
                raise ValidationError('File too big.')

    def makeDoc(self):
        doc = Document()
        doc.title = self.cleaned_data.get('docTitle')
        if self.cleaned_data.get('docMethod') == 'filePanel':
            doc.docfile = self.cleaned_data.get('docFile')
        else:
            fpath = default_storage.get_available_name('pasteUpload.txt')
            savedDoc = default_storage.save(fpath, ContentFile(self.cleaned_data.get('docText')))
            doc.docfile = savedDoc
        return doc
