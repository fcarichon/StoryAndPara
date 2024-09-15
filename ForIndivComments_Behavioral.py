import spacy
import emoji
import language_tool_python


class behavior():
    
    def __init__(self): 
        
        self.nlp = spacy.load("en_core_web_sm")
        self.tool = language_tool_python.LanguageTool('en-US')
        self.emoji_behav_codes = ["1F600","1F603","1F604","1F601","1F606","1F605","1F923","1F602","1F642","1F643","1FAE0","1F609","1F60A","1F607","1F970","1F60D","1F929","1F618","1F617",
                    "263A","1F61A","1F619","1F972","1F60B","1F61B","1F61C","1F92A","1F61D","1F911","1F917","1F92D","1FAE2","1FAE3","1F92B","1F914","1FAE1","1F910","1F928",
                     "1F636","1F32B","1F60F","1F612","1F644","1F62C","1F62E","1F4A8","1F925","1FAE8","1F642","2194",
                     "1F642","2195","1F60C","1F614","1F62A","1F924","1F634","1F637","1F912","1F915","1F922","1F92E","1F927","1F975","1F976","1F974","1F635","1F635",
                     "1F4AB","1F973","1F615","1FAE4","1F61F","1F641","2639","1F62E","1F62F","1F632","1F633","1F97A","1F979","1F626","1F627","1F628","1F630","1F625",
                     "1F622","1F62D","1F631","1F616","1F623","1F61E","1F613","1F629","1F62B","1F971","1F624","1F621","1F620","1F92C","1F63A","1F638","1F639","1F63B",
                     "1F63C","1F63D","1F640","1F63F","1F63E","1F648","1F649","1F64A","1F44B","1F44B","1F3FB","1F44B","1F3FC","1F44B","1F3FD","1F44B","1F3FE","1F44B",
                     "1F91A","1F91A","1F91A","1F91A","1F91A","1F91A","1F590", "270B","1F596","1FAF1","1FAF2","1FAF3","1FAF4","1FAF7","1FAF8","1F44C",
                     "1F90C","1F90F","270C","1F3FF", "1F91E","1FAF0","1F91F","1F918","1F919","1F448","1F449","1F446","1F595","1F447","1FAF5","261D","1F44D","1F44E","270A",
                     "1F44A","1F91B","1F91C", "1F44F","1F64C","1FAF6","1F450","1F932","1F91D","1FAF1","200D","1FAF2","1F64F","270D","1F485","1F933","1F4AA","1F64D","2642",
                     "1F64E","1F645","1F646","1F481","1F64B","1F9CF","1F>647","1F926","1F937","1F486","1F487","1F6B6","1F6B6","1F9CD","1F9CE","27A1","1F9D1","1F9AF",
                     "1F9D1","1F468","1F469","1F9D1","1F3C3","2640","1F483","1F57A","1F574","1F46F","1F9D6","1F9D7","1F93A","1F3C7","26F7","1F3C2","1F3CC","1F3C4",
                     "1F6A3","1F3CA","26F9","1F3CB","1F6B4","1F6B5","1F938","1F93C", "1F93D","1F93E","1F939","1F9D8","1F6C0","1F6CC"]
        
        self.slangs_behav_list = ["lol", "lmao", "rofl", "rotfl", "roflmao", "roflcopter", "haha", "hehe", "xd", "huehuehue", "lmfao", "smh", "teehee", "flex", "Snort", "chortle", "bwahaha", "Pmsl", "lolz", 
        "bwg", "brb", "ttyl", "lulw","lulz", "lololol", "omegalul", "trololol", "lolwut", "j4g", "kekekek", "kekek", "kek", "lqtm", "lurk", "facepalm", "555", "asg", "jajaja", "jejeje",
        "mdr", "mkm", "xpldr", "ptdr", "rsrsrs", "qq"]
        
        self.intent_list = ["will", "wish", "would", "should", "could", "intend", "plan", "aspire", "aim", "envisage", "strive", "contemplate", "try", "attempt"]
        self.third_sing = ['he','she', 'him', 'her', 'his', 'hers']
        
    def behav_emoji(self, comments_list):
    
        behav_reac = []
        reac_comment = []
        for comment in comments_list:
            comment_reac = 0
            reac = 0
            comment_tok = [tok.text.lower() for tok in self.nlp.tokenizer(str(comment))]
            for token in comment_tok:
                #detect if token is an emoji
                emoji_ = bool(emoji.emoji_count(token))
                if emoji_:
                    try:
                        unicode = '{:X}'.format(ord(token))
                        #print('{:X}'.format(ord(token)), unicode)
                        if unicode in self.emoji_behav_codes:
                            comment_reac += 1
                            reac = 1
                    except:
                        pass
            behav_reac.append(comment_reac)
            reac_comment.append(reac)

        return behav_reac, reac_comment
    
    def nb_comments(self, comments_list):
        return len(comments_list)
    
    def behav_slangs(self, comments_list):
    
        behav_reac = []
        for comment in comments_list:
            nb_reac = 0
            comment_tok = [tok.text.lower() for tok in self.nlp.tokenizer(str(comment))]
            for token in comment_tok:
                if token in self.slangs_behav_list:
                    nb_reac += 1
            behav_reac.append(nb_reac)

        return behav_reac
        
    def error_type(self, comments_list):
    
        mistakes = []
        grammar_list = []
        for comment in comments_list:
            grammar_mistake = 0
            list_mistakes = self.tool.check(comment)
            for mistake in list_mistakes:
                try:
                    category = mistake["category"]
                    if category == 'GRAMMAR':
                        grammar_mistake = 1
                except:
                    pass

            mistakes.append(len(list_mistakes))
            grammar_list.append(grammar_mistake)

        return mistakes, grammar_list
    
    def take_time_to_write(self, comment_list, user_name, nb_words = 20, nb_charac=80):
    
        take_time = []
        for comment in comment_list:
            cite, size = False, False
            token_list = [tok.text.lower() for tok in self.nlp.tokenizer(str(comment))]
            for token in token_list:
                if user_name in token:
                    cite = True
                if token in self.third_sing:
                    cite = True

            sentence_size = len(token_list)
            charac_size = len(comment)
            if sentence_size > nb_words: 
                size = True
            if charac_size > nb_charac:
                size = True

            if cite and size:
                take_time.append(1)
            else:
                take_time.append(0)

        return take_time
    
    def answer(self, comment_list, user_name):
    
        answer_ = []
        for comment in comment_list:
            cite = False
            count_user = 0
            token_list = [tok.text.lower() for tok in self.nlp.tokenizer(str(comment))]
            for token in token_list:
                if user_name in token:
                    cite = True
                    count_user += 1
                if token in self.third_sing:
                    cite = True

            count_a = comment.count('@')
            if cite and count_a >= count_user:
                answer_.append(1)
            else:
                answer_.append(0)

        return answer_
    
    def intent(self, comment_list):
    
        intent_ = []
        for comment in comment_list:
            count_intent = 0
            token_list = [tok.text.lower() for tok in self.nlp.tokenizer(str(comment))]
            for token in token_list:
                if token in self.intent_list:
                    count_intent += 1

            intent_.append(count_intent)

        return intent_