"""
Pre-trained models (NLTK, Spacy, StanfordCoreNlP)
"""
import nltk
import spacy
from pycorenlp import StanfordCoreNLP


def ie_nltk(txt):
    """
    Name entity recognition (NLTK)
    """
    print(txt)
    # tokenizing text
    tok = nltk.word_tokenize(txt)
    # part of speech tagging
    pos_tag = nltk.pos_tag(tok)
    # Name entity recognition
    chunks = nltk.ne_chunk(pos_tag)
    ent = []
    for item in chunks:
        if hasattr(item, 'label'):
            ent.append((' '.join(ch[0] for ch in item), item.label))
    print(ent)


def ie_spacy(txt):
    """
        Name entity recognition (Spacy)
    """
    nlp_en = spacy.load("en_core_web_lg")
    tok = nlp_en(txt)
    ent = []
    for items in tok.sents:
        items = items.string.strip()
        item = nlp_en(items)
        ent.append([i.text, i.label_] for i in item.ents)
    for items in ent:
        for item in items:
            print(item)


def ie_stanford(txt):
    """
        Name entity recognition (StanfordCoreNlP)
    """
    nlp = StanfordCoreNLP('http://localhost:9000')
    ann = nlp.annotate(txt, properties={
        'annotators': 'tokenize, ssplit, pos, lemma, ner',
        'outputFormat': 'json'})
    ent = [(w['word'], w['ner']) for sent in ann["sentences"] for w in sent['tokens']]
    for item in ent:
        print(item[0], item[1])