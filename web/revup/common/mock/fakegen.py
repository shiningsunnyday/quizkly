import random
import time
from ..question import Question
import logging

logger = logging.getLogger(__name__)

lorem_ipsum_ans = [
    "Lorem ipsum dolor sit amet",
    "consectetur adipiscing elit",
    "sed do eiusmod tempor incididunt",
    "ut labore et dolore magna aliqua",
    "Ut enim ad minim veniam",
    "quis nostrud exercitation ullamco",
    "laboris nisi ut aliquip",
    "ex ea commodo consequat.",
    "Duis aute irure dolor in",
    "reprehenderit in voluptate velit",
    "esse cillum dolore eu",
    "fugiat nulla pariatur.",
    "Excepteur sint occaecat",
    "cupidatat non proident",
    "sunt in culpa qui officia deserunt",
    "mollit anim id est laborum",
    "Sed ut perspiciatis",
    "unde omnis iste",
    "natus error sit voluptatem",
    "accusantium doloremque laudantium",
    "totam rem aperiam",
    "eaque ipsa quae ab illo",
    "inventore veritatis et quasi architecto",
    "beatae vitae dicta sunt explicabo"
]

num_random_dist = 4
class FakeGen:
    _settings = None
    def __init__(self, settings):
        self._settings = settings

    def get_question(self, sentence, previous_sentences=None):
        if self._settings['ERROR_PROB_RANGE'] > 0:
            if random.randint(0, self._settings['ERROR_PROB_RANGE'] - 1) == 0:
                logger.info("Making error")
                raise ValueError("random failure")
        random.shuffle(lorem_ipsum_ans)
        context_sentence = ""
        if previous_sentences:
            context_sentence = previous_sentences[-1]
        time.sleep(self._settings['DELAY'])
        return Question(context_sentence+sentence, lorem_ipsum_ans[0], lorem_ipsum_ans[1:num_random_dist+1])
