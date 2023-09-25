import numpy as np

def sort_found_objects(found_objects, cats):
    """Sort the found objects by the order of the categories"""
    found_objects = sorted(found_objects, key=lambda x: cats.index(x))
    return found_objects


def get_word_diversity(planner, word, candidates):
    dis = []
    doc1 = planner.nlp(word)
    for word in candidates:
        doc2 = planner.nlp(word)
        dis.append(doc1.similarity(doc2))
    return candidates[np.argmax(np.asarray(dis))]