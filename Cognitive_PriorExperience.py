import spacy 
from nltk.tokenize import sent_tokenize, word_tokenize



class experience():
    
    def __init__(self, first_sing = ['i', 'myself'], first_poss = ['my', 'me', 'mine'], third_sing = ['he', 'his','her','she', 'him', 'somebody', 'himself', 'herself', 'hers'], 
                past_tense = ["VBD", "VBN"], valid_ent = ["PERSON"]): 
        
        self.nlp = spacy.load("en_core_web_sm")
        self.first_sing = first_sing
        self.first_poss = first_poss
        self.third_sing = third_sing
        self.past_tense = past_tense
        self.sub_coord_list = ["But", "however", "so", "then", "thus", "therefore", "because", "accordingly", "consequently", "hence",
        "as a result of", "as long as", "as things go", "being", "by cause of", "by reason of", "by virtue of", "considering", "due to"
        "nonetheless", "notwithstanding", "yet", "after all", "all the same", "anyhow", "be that as it may", "for all that"
        "after", "although", "as", "as if", "as long as", "before", "despite", "even if", "even though", "if", "in order that", 
        "rather than", "since", "so that", "that", "though", "unless", "until", "when", "where", "whereas", "whether", "while",
        "provided that", "in order to", "whenever", "nor",  "than"]
        self.valid_ent = valid_ent
        
    def named_entities(self, comment_list):
        """ Once again level = post -- we average for all comments here to keep persistence with previous metrics
        # Présence d’une entité nommée ou d’un pronom, suivi de l’utilisation du passé, avec une distinction sur le pronom 
        #(soit « je » + passé, soit « tu/il » + possessif 1e personne + passé) : À quel point le commentateur se rappelle 
        #d’une expérience passée dans son commentaire ?
        """
        rememberance = []
        for comment in comment_list:
            remember = 0
            ent_count, first_count, fposs_count, third_count, past_count = False, False, False, False, False,
            sent_list = sent_tokenize(comment)
            for sent in sent_list:
                doc = self.nlp(sent)
                for ent in doc.ents:
                    if ent.label_ in self.valid_ent:
                        ent_count = 1
                for token in doc:
                    token_text = token.text.lower()
                    if token.pos_ == "PRON" and token_text in self.first_sing:
                        first_count = True
                    if token.pos_ == "PRON" and token_text in self.first_poss:
                        fposs_count = True
                    if token.pos_ == "PRON" and token_text in self.third_sing:
                        third_count = True
                    if token.tag_ in self.past_tense:
                        past_count = True

                if first_count and past_count:
                    remember = 1
                if third_count and third_count and past_count:
                    remember = 1

            rememberance.append(remember)

        avg_remember = sum(rememberance) / len(comment_list)

        return avg_remember
    
    
    def causal_subord(self, comments_list):
        """#détection d’une conjonction de subordination (e.g., comme, quand, afin que, parce que, etc.) et la présence d’une entité 
        nommée dans la clause de subordination : Est-ce que le commentateur évoque une situation impliquée par la publication ?"""
        
        implied_list = []
        for comment in comments_list:
            sentence_list = sent_tokenize(comment)
            count_implied = 0
            for sentence in sentence_list: 
                sentence_tok = [tok.text for tok in self.nlp.tokenizer(str(sentence))]
                idx_list = []
                for conjonction in self.sub_coord_list:
                    if conjonction in sentence_tok:
                        idx_list.append(sentence_tok.index(conjonction))
                        
                if len(idx_list) != 0:
                    #Getting the inedex of the first clause in the sentence - consider as main
                    final_idx = min(idx_list)
                    #Creating the sentence from the rest of the token
                    clause_sent = (" ").join(sentence_tok[final_idx:])
                
                    doc_clause = self.nlp(clause_sent)
                    for token in doc_clause:
                        token_text = token.text.lower()
                        if token.pos_ == "PRON" and token_text in self.third_sing:
                            count_implied = 1
                    for ent in doc_clause.ents:
                        if ent.label_ in self.valid_ent:
                            count_implied = 1
                            
            implied_list.append(count_implied)
            
        avg_implication= sum(implied_list) / len(comments_list)
        
        return avg_implication
    
    def EN_compare(self, comments_list):
        """
        #présence d’au moins deux entités nommées (ou pronoms) différentes dans un span donné de phrases consécutives :
        #Est-ce que le commentateur compare la marque-personne à une autre personne ?
        """
        comments_compare = 0
        for comment in comments_list:
            count_compare = 0
            max_compare = 0
            sentence_list = sent_tokenize(comment)
            for sentence in sentence_list:
                compare_sentence = 0
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    if ent.label_ in self.valid_ent:
                        compare_sentence = 1
                for token in doc:
                    token_text = token.text.lower()
                    if token.pos_ == "PRON" and token_text in self.third_sing:
                        compare_sentence = 1

                count_compare += compare_sentence

                if count_compare > max_compare:
                    max_compare = count_compare
            if max_compare >= 2:
                comments_compare += 1

        avg_comparison = comments_compare / len(comments_list)

        return avg_comparison