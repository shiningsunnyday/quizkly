class Question(object):
    """
    Object to store a question
    """
    def __init__(self, qn_text, ans, distractors):
        self.qn_text = qn_text
        self.ans = ans
        self.distractors = distractors

    def __str__(self):
        return 'Qn = %s \n Ans = %s \n Distractors = %s' % (self.qn_text, self.ans, 
                                                            self.distractors) 
