from __future__ import absolute_import
from django.conf import settings
from django.db import transaction
from celery import shared_task, Task, chord
from .models import Document, Question, Distractor
from itertools import repeat
from traceback import format_exc
from celery.utils.log import get_task_logger
from celery.signals import worker_process_init
from concurrent.futures import ThreadPoolExecutor

logger = get_task_logger(__name__)

executor = ThreadPoolExecutor(max_workers=1)
generator = None

def load_gen():
    logger.info('Initializing generator')
    gen = settings.REVUP_MKGEN()
    logger.info('Generator initialized')
    return gen

@worker_process_init.connect
def init_tasks(**kwargs):
    global generator
    generator = executor.submit(load_gen)

# @shared_task decorator allows the task to bind to any celery app instance, so we don't define app here
@shared_task
# generate question, based on sentence, and save as qn_id under document doc_id
def generate_questions(doc_id, qn_id, sentence, previous_sentences):
    try:
        doc = Document.objects.get(id=doc_id)
        if doc.status == Document.IN_QUEUE:
            doc.status = Document.PROCESSING
            doc.save()

        logger.debug("doc {0} qn {1} started".format(doc_id, qn_id))
        gen = generator.result()
        qn = gen.get_question(sentence, previous_sentences)

        if qn is None:
            logger.warning("QuestionGenerator returned None.")
            return True

        with transaction.atomic():  # TODO: Is atomicity necessary on Read Commited?
            qo = Question(doc=doc, qnNo=qn_id, qnText=qn.qn_text, ans=qn.ans)
            qo.save()
            for dist in qn.distractors:
                ds = Distractor(qn=qo, distractor=dist)
                ds.save()
        logger.debug("doc {0} qn {1} done".format(doc_id, qn_id))
        return True
    except:
        # Celery won't let us catch errors from the chord callback, so we catch manually
        # TODO: Alternative solution is to attach error callbacks manually to each task
        logger.error(format_exc())
        return False

@shared_task
#start splitting document into sentences and send off to process
def process_doc(doc_id):
    doc = Document.objects.get(id=doc_id)
    from .preprocess import preprocess_sentences
    #get_sentences_from_file
    #sentences = get_sentences_from_file(doc.docfile)
    sentences = preprocess_sentences(str(doc.docfile.read()))
    previous_sentences = [sentences[max(i-5, 0):i] for i in range(len(sentences))]
    tasks = generate_questions.chunks(zip(repeat(doc_id), 
                                          range(0, len(sentences)), 
                                          sentences, 
                                          previous_sentences),
                                      settings.REVUP_CHUNK_SIZE).group()
    callback = finish_doc.s(doc_id)
    # FIXME: Error when tasks array is empty
    # NOTE: chord does not unlock on RabbitMQ (bug https://github.com/celery/celery/issues/2725)
    chord(tasks)(callback)

@shared_task(bind=True)
# Called after all sentences have been processed, to set document status
# apparently .s(x,y,...) binds x,y,... AFTER self (reasonable) and result (WTF?)
def finish_doc(self, result, doc_id):
    doc = Document.objects.get(id=doc_id)

    # FIXME: What if res is empty? Doc status is ERROR
    # FIXME: Logger will not fire outside of celery worker (i.e. in django)
    if all(result):
        logger.debug("Doc {0} finished")
        doc.status = Document.COMPLETE
    else:
        logger.debug("Doc {0} failed")
        doc.status = Document.ERROR
    doc.save()

