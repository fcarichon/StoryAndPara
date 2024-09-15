import spacy 
from nltk.tokenize import sent_tokenize


class construction():
    
    def __init__(self, first_sing = ['i', 'myself', 'my', 'me', 'mine'], first_plur = ['we', 'our','us'], second = ['you', 'your', 'u', 'ya', 'yours', 'ur'], 
                 third_sing = ['he','she', 'him', 'her', 'his', 'hers']): 
        
        self.first_sing = first_sing
        self.first_plur = first_plur
        self.second = second
        self.third_sing = third_sing
        self.authorized_pron = second + third_sing
        self.personal_pron = first_sing + first_plur
        self.nlp = spacy.load("en_core_web_sm")
        self.sub_coord_list = ["But", "however", "so", "then", "thus", "therefore", "because", "accordingly", "consequently", "hence",
        "as a result of", "as long as", "as things go", "being", "by cause of", "by reason of", "by virtue of", "considering", "due to"
        "nonetheless", "notwithstanding", "yet", "after all", "all the same", "anyhow", "be that as it may", "for all that"
        "after", "although", "as", "as if", "as long as", "before", "despite", "even if", "even though", "if", "in order that", 
        "rather than", "since", "so that", "that", "though", "unless", "until", "when", "where", "whereas", "whether", "while",
        "provided that", "in order to", "whenever", "nor",  "than"]
        
    def context_proj(self, username, comments_list):
    
        
        relation_list=[]
        username = username.lower()
        for comments in comments_list:
            relation = 0
            sentence_list = sent_tokenize(comments)
            for sentence in sentence_list:
                quote = False
                first = False
                doc = self.nlp(sentence)
                for token in doc:
                    token_text = token.text.lower()
                    if username in token_text:
                        quote = True
                    if token_text in self.authorized_pron:
                        quote = True
                    if token_text in self.personal_pron:
                        first = True

                if quote and first:
                    relation = 1
            relation_list.append(relation)

        avg_relation = sum(relation_list)/len(comments_list)

        return avg_relation
    
    
    def obsession(self, username, comments_list):
    
        obssess_list = []
        for comments in comments_list:
            count_norm = 0
            token_list = [tok.text.lower() for tok in self.nlp.tokenizer(str(comments))]
            for token in token_list:
                if username in token:
                    count_norm += 1/len(token_list)
                if token in self.authorized_pron:
                    count_norm += 1/len(token_list)
            obssess_list.append(count_norm)

        avg_obsess = sum(obssess_list)/len(comments_list)

        return avg_obsess
    
    def action_link(self, username, comments_list):
    
        action_list = []
        for comment in comments_list:
            sentence_list = sent_tokenize(comment)
            bin_action = 0
            for sentence in sentence_list:
                first = False
                user_ = False

                sentence_tok = [tok.text for tok in self.nlp.tokenizer(str(sentence))]
                #Checking if first pronoun is used
                for elem in self.personal_pron:
                    if elem in sentence_tok:
                        first=True
                #Then if subordination conjonction and username or pronouns
                if first:
                    idx_list = []
                    for conjonction in self.sub_coord_list:
                        if conjonction in sentence_tok:
                            idx_list.append(sentence_tok.index(conjonction))

                    #Being sure we have a clause
                    if len(idx_list)>0:
                        #Getting the inedex of the first clause in the sentence - consider as main
                        final_idx = min(idx_list)
                        #Creating the sentence from the rest of the token
                        clause_sent = (" ").join(sentence_tok[final_idx:])
                        doc_clause = self.nlp(clause_sent)

                        for token in doc_clause:
                            token_text = token.text.lower()
                            if username in token_text or token_text in self.authorized_pron:
                                user_ = True
                        if user_:
                            bin_action = 1
            action_list.append(bin_action)
        avg_action = sum(action_list)/len(comments_list)

        return avg_action
    
    def direct_mention(self, username, comments_list):
    
        """count the number of commentaire with direct mention of the user"""

        count_ = 0
        mention_name = "@" + username
        for comment in comments_list:
            mention = False
            doc = self.nlp(comment)
            for token in doc:
                token_text = token.text.lower()
                if mention_name in token_text:
                    mention = True
            if mention:
                count_+=1

        return count_/len(comments_list)