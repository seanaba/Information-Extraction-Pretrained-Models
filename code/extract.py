# -*- coding: utf-8 -*-
"""
Information extraction (pre-trained models)
"""
import models
import os
import io


class ExtractDoc(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def get_data(self):
        """
        get all documents
        """
        docs = [os.path.join(self.data_path, doc) for doc in os.listdir(self.data_path)
                if os.path.isfile(os.path.join(self.data_path, doc))]
        return docs

    def extract_info(self):
        """
        Information extraction (pre-trained models)
        1- Collect  documents
        2- Extract information
        """
        # loads all documents
        docs = self.get_data()
        # extracts information using pre-trained models
        for doc in docs:
            with io.open(doc, "r") as f_in:
                text = f_in.read()
            models.ie_nltk(text)
            models.ie_spacy(text)
            models.ie_stanford(text)

