#Import des tokenizers - sent_tokenize pour les phrases et word_tokenize pour les mots avec nltk
from nltk.tokenize import sent_tokenize
import spacy 


class anticipation():
    
    def __init__(self, second = ['you', 'your', 'u', 'ya', 'yours', 'ur'], third_sing = ['he','she', 'him', 'her', 'his', 'hers'], conditional_terms = ["could", "might", "wish", "would", "if"],
                valid_pos = ["MD", "VBG"]): 
        
        self.second = second
        self.second = third_sing
        self.conditional_terms = conditional_terms
        self.authorized_pron = second + third_sing
        self.valid_pos = valid_pos
        self.nlp = spacy.load("en_core_web_sm")
        
        
    #Evolution de la marque
    def brand_evo(self, username, comments_list):

        evo_list = []
        username = username.lower()
        for comments in comments_list:
            evolution = 0
            sentence_list = sent_tokenize(comments)
            for sentence in sentence_list:
                quote = False
                future = False
                doc = self.nlp(sentence)
                for token in doc:
                    token_text = token.text.lower()
                    if username in token_text:
                        quote = True
                    if token_text in self.authorized_pron:
                        quote=True
                    if token.tag_ in self.valid_pos:
                        future = True
                if quote and future:
                    evolution = 1
            evo_list.append(evolution)

        return evo_list
    
    def brand_cond(self, username, comments_list):
    
        condit_list = []
        username = username.lower()
        for comments in comments_list:
            conditional = 0
            sentence_list = sent_tokenize(comments)
            for sentence in sentence_list:
                quote = False
                condition = False
                doc = self.nlp(sentence)
                for token in doc:
                    token_text = token.text.lower()
                    if username in token_text:
                        quote = True
                    if token_text in self.authorized_pron:
                        quote=True
                    if token_text in self.conditional_terms:
                        condition = True

                if quote and condition:
                    conditional = 1
            condit_list.append(conditional)

        return condit_list