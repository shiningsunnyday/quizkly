import spacy

AMBIGUOUS_DETERMINERS = [u'this', u'that', u'these', u'such', u'its']

def dep_context(prev_sents, current_doc, parser):
    """
    Returns a context sentence from prev_sents if spacy doc has unresolved coreferences
    """
    if not is_doc_ambiguous(current_doc):
        return ""
    else:
        #return the closest non-ambiguous sentence from prev_sents
        context = ""
        for sent in reversed(prev_sents):
            context = sent + context
            if not is_sent_ambiguous(sent, parser):
                return context
        return ""

def is_sent_ambiguous(sent, parser):
    """
    Checks if sent is ambiguous
    """
    doc = parser(sent)
    return is_doc_ambiguous(doc)
    

def is_doc_ambiguous(doc):
    """
    Check if a spacy doc is ambiguous
    """
    subj = 0
    # check if sentence starts with a prep or modifier 
    # also, make sure it doesnt link to an ambiguous det
    if doc[0].dep_ == 'prep' or (doc[0].dep_[-3:] == 'mod' and doc[0].is_stop):
        subtree = list(doc[0].subtree)
        if len(subtree) == 0:
            return True
        if ambig_subtree_checker(subtree): 
            return True
    for word in doc:
        if word.dep_[0:5] == 'nsubj':
            subj += 1
            if word.pos_ == 'PRON':
                return True #if the subj is a pronoun, and is not preceded by a specific subj 
            if ambig_subtree_checker(word.subtree): 
                return True 
            #once we come across a concrete subject, we are confident the sentence is unambiguous
            break
        elif word.dep_[0:4] == 'pobj' and subj == 0:
            if ambig_subtree_checker(word.subtree):
                return True 
    return False


def ambig_subtree_checker(subtree):
    """
    Checks if any of the leafs in a subtree is an anbiguous determiner
    """
    for leaf in subtree:
        if leaf.orth_.lower() in AMBIGUOUS_DETERMINERS:
            return True 
    return False

if __name__ == "__main__":
    PARSER = spacy.load('en', add_vectors=False)
    PREV_SENTS = ["The tricuspid valve has three cusps, which connect to chordae tendinae" \
                   "and three papillary muscles named the anterior, posterior, and septal" \
                   "muscles, after their relative positions.", 
                  "The mitral valve lies between the left atrium and left ventricle.", 
                  "It is also known as the bicuspid valve due to its having two cusps, " \
                    "an anterior and a posterior cusp."]
    SENT = "It is the sequence of these four nucleobases along the backbone " \
            "that encodes biological information."
    DOC = PARSER(unicode(SENT))
    print(dep_context(PREV_SENTS, DOC, PARSER))
