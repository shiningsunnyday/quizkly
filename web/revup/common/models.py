from django.db import models
from django.conf import settings
from django.utils import timezone

class RevupUserInfo(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    school = models.CharField(max_length=200, blank=True)

class Document(models.Model):
    IN_QUEUE = 0
    PROCESSING = 1
    COMPLETE = 2
    ERROR = -1
    STATUS_CHOICES = (
        (IN_QUEUE, 'In Queue'),
        (PROCESSING, 'Processing'),
        (COMPLETE, 'Complete'),
        (ERROR, 'Error')
    )

    docfile = models.FileField(upload_to='documents/')
    status = models.SmallIntegerField(choices=STATUS_CHOICES, default=0)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='documents', on_delete=models.CASCADE)

    title = models.CharField(max_length=64, default='')
    created_time = models.DateTimeField(default=timezone.now, editable=False, blank=True)
    modified_time = models.DateTimeField(default=timezone.now, editable=False, blank=True)

    def save(self, *args, **kwargs):
        # update modified date on save
        if kwargs.pop('update_modified',False):
            self.modified = timezone.now()
        return super(Document, self).save(*args,**kwargs)

    def get_absolute_url(self):
        from django.urls import reverse

        return reverse('document', args=[str(self.id)])


class Question(models.Model):
    doc = models.ForeignKey(Document, related_name='questions', on_delete=models.CASCADE)
    qnText = models.CharField(max_length=256)
    ans = models.CharField(max_length=64)
    qnNo = models.IntegerField()
    class Meta:
        ordering = ['qnNo']

class Distractor(models.Model):
    qn = models.ForeignKey(Question, related_name='distractors', on_delete=models.CASCADE)
    distractor = models.CharField(max_length=64)
    def __unicode__(self):
        return self.distractor


