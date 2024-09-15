import spacy 
from collections import defaultdict 
from nltk.tokenize import sent_tokenize, word_tokenize
import statistics
import numpy as np
from numpy.linalg import norm

class Chronology():
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        
        self.future = ["next day", "following day", "day after", "next morning", "following morning", "tomorrow", "afterward", "shortly", "soon", "later", 
                       "upcoming", "in the coming days", "in the future", "down the road", "forthcoming", "impending"]
        self.past = ["bygone", "past", "foretime", "last day", "yesterday", "not long ago", "recently", "other day", "previous", "prior", "formerly", 
                     "earlier", "back then", "yesteryear", "once", "once upon a time"]
        self.present = ["today", "this day", "this very day", "before tomorrow", "this morning", "this afternoon", "this evening", "present-day", 
                        "present day", "current", "now", "here and now", "present moment", "the time being", "at present", "at the present time", 
                        "at the moment", "in the moment", "currently", "right now", "ongoing"]
    
    def tokenize(self, text):
        return [tok.text for tok in self.nlp.tokenizer(str(text))]

    def bin_date(self, text):
        """ Creating a binary variable if we detect a date in a sentence"""
        doc_date = self.nlp(text)
        for entity in doc_date.ents:
            if entity.label_ in ["DATE", "TIME"]:
                return 1
        
        return 0
    
    def logic_order(self, list_of_tense):
        
        logic_chrono = 0
        ordered_list = list(dict.fromkeys(list_of_tense))
        if len(ordered_list) == 1:
            logic_chrono = 1
        if len(ordered_list) == 2:
            if ordered_list in [["past", "future"], ["past", "present"], ["present", "future"]]:
                logic_chrono = 1
        if len(ordered_list) == 3:
            if ordered_list == ["past", "present", "future"]:
                logic_chrono = 1
        
        return logic_chrono
        
    def special_chrono(self, text):
        """ Create a binary variable to see if text include today, yesterday, or tomorrow. Create a binary variable to check if it follows
        a logical order
            Create a variable with the proportion of element include in those lists"""

        detect_time = []
        all_time = self.future + self.past + self.present
        for elem in all_time:
            if elem in text:
                if elem in self.future:
                    detect_time.append("future")
                if elem in self.present:
                    detect_time.append("present")
                if elem in self.past:
                    detect_time.append("past")
        
        logic_chrono = self.logic_order(detect_time)
                
        return len(detect_time)/len(self.tokenize(text)), logic_chrono
    
    def tense(self, text):
        """ Create a binary variable to see if tenses are used in correct order"""
        
        tag_list = []
        sent_list = sent_tokenize(text)
        for sent in sent_list:
            doc = self.nlp(sent)
            count_past = 0
            count_present = 0
            count_future = 0
            for token in doc:
                if token.tag_ in ["VBD", "VBN"]:
                    count_past +=1 
                if token.tag_ in ["MD", "VBG"]:
                    count_future += 1
                if token.tag_ in ["VBZ", "VBP"]:
                    count_present += 1
            #Order is crucial here because argmax return first value and if we detect both past and future, or both future and present they are to privilege    
            count_list = np.array([count_past, count_future, count_present])
            index = count_list.argmax()
            if index == 0:
                tag_list.append("past")
            elif index == 1:
                tag_list.append("future")
            else:
                tag_list.append("present")
                
        logic_tense = self.logic_order(tag_list)
        return logic_tense