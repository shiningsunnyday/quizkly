import kenlm

def scoreword(model, words, history):
    """
    Score a word on how well it fits into its history
    """

    if history:
        query = history + words
    else:
        query = words
 
    state = kenlm.State()
    next_state = kenlm.State()

    if len(query) > model.order:
        query = query[-model.order:]
        model.NullContextWrite(state)
    else:
        model.BeginSentenceWrite(state)

    total = 0
    for word in query:
        total += model.BaseScore(state, word, next_state)
        state, next_state = next_state, state
        
    return total

