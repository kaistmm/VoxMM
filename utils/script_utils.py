import os, glob, shutil, argparse, yaml, re, json

PATTERN = r'![^!]*!|\{[^}]*\}|\([^/][^)]*\)|\[[^\]]*\]|<[^>]*>|[\w\'.]+'


class Script_Generator():
    def __init__(self,capitalize=True,
                    apostrophe=True,
                    hypen=False,
                    numeric_format="pronunciation", # digit, pronunciation
                    space_on_abbreviation=False, # give space between each alphabet if True
                    default_inaudible_process="drop", #drop, token
                    default_uncertain_process="drop", #word, drop, token
                    default_diffluency_process="drop", #word, drop, token
                    default_interjection_process="drop", #word, drop, token
                    interjection_to_word=[], # exception interjection list to convert into word
                    interjection_to_token=[], # exception interjection list to convert into token
                    interjection_to_drop=[], # exception interjection list to drop
                    inaudible_token="<unknown>", # use only if default_inaudible_process=token
                    uncertain_token="<unknown>", # use only if default_uncertain_process=token
                    diffluency_token="<diffluency>", # use only if default_diffluency_process=token
                    interjection_token="<interjection>", # use only if default_interjection_process=token
                    **kwargs
                    ):
        self.capitalize = capitalize
        self.apostrophe = apostrophe
        self.hypen = hypen
        self.num_format = numeric_format
        self.abb_space = space_on_abbreviation

        self.inaud_proc = default_inaudible_process
        self.uncert_proc = default_uncertain_process
        self.diff_proc = default_diffluency_process
        self.interj_proc = default_interjection_process

        self.interj2word = interjection_to_word if isinstance(interjection_to_word,list) else interjection_to_word.replace("[","").replace("]","").replace(" ","").split(',')
        self.interj2token = interjection_to_token if isinstance(interjection_to_token,list) else interjection_to_token.replace("[","").replace("]","").replace(" ","").split(',')
        self.interj2drop = interjection_to_drop if isinstance(interjection_to_drop,list) else interjection_to_drop.replace("[","").replace("]","").replace(" ","").split(',')

        self.interj2word = [w.strip().lower() for w in self.interj2word]
        self.interj2token = [w.strip().lower() for w in self.interj2token]
        self.interj2drop = [w.strip().lower() for w in self.interj2drop]

        self.inaud_token = inaudible_token
        self.uncert_token = uncertain_token
        self.diff_token = diffluency_token
        self.interj_token = interjection_token
        self.tokens = [inaudible_token, uncertain_token, diffluency_token, interjection_token]


    def numeric_process(self, txt):
        txt_splitted = re.findall(PATTERN, txt)
        for idx, inst in enumerate(txt_splitted):
            if '/' in inst:
                inst = inst.replace('(','').replace(')','')
                if self.num_format=='pronunciation':
                    txt_splitted[idx] = inst.split('/')[1].strip()
                elif self.num_format=='digit':
                    txt_splitted[idx] = inst.split('/')[0].strip()
                else:
                    raise Exception(f'Invalid numeric format: {self.num_format}')

        return ' '.join(txt_splitted)

    def abbreviation_process(self, txt):
        txt_splitted = re.findall(PATTERN, txt)
        for idx, inst in enumerate(txt_splitted):
            if '!' in inst:
                inst = inst.replace('!','').strip()
                txt_splitted[idx] = ' '.join(list(inst)) if self.abb_space else inst

        return ' '.join(txt_splitted)

    def inaudible_process(self, txt):
        if self.inaud_proc=='drop':
            return txt.replace('(inaudible)','').strip()

        elif self.inaud_proc=='token':
            return txt.replace('(inaudible)',self.inaud_token).strip()

        else:
            raise Exception(f'Invalid inaudible process setting: {self.inaud_proc}')

    def uncertain_process(self, txt):
        txt_splitted = re.findall(PATTERN, txt)
        new_txt = []
        for inst in txt_splitted:
            if '[' in inst:
                inst = inst.replace('[','').replace(']','').strip()
                if self.uncert_proc=='word':
                    new_txt.append(inst)

                elif self.uncert_proc=='drop':
                    continue

                elif self.uncert_proc=='token':
                    new_txt.append(self.uncert_token)

                else:
                    raise Exception(f'Invalid uncertain process setting: {self.uncert_proc}')

            else:
                new_txt.append(inst)

        return ' '.join(new_txt)


    def diffluency_process(self, txt):
        txt_splitted = re.findall(PATTERN, txt)
        new_txt = []
        for inst in txt_splitted:
            if '<' in inst and not inst in self.tokens:
                inst = inst.replace('<','').replace('>','').strip()
                if self.diff_proc=='word':
                    new_txt.append(inst)

                elif self.diff_proc=='drop':
                    continue

                elif self.diff_proc=='token':
                    new_txt.append(self.diff_token)

                else:
                    raise Exception(f'Invalid difflency process setting: {self.diff_proc}')

            else:
                new_txt.append(inst)

        return ' '.join(new_txt)


    def interjection_process(self, txt):
        txt_splitted = re.findall(PATTERN, txt)
        new_txt = []
        for inst in txt_splitted:
            if '{' in inst:
                inst = inst.replace('{','').replace('}','').strip().lower()
                if inst in self.interj2word:
                    new_txt.append(inst)

                elif inst in self.interj2drop:
                    continue

                elif inst in self.interj2token:
                    new_txt.append(self.interj_token)

                else:
                    if self.interj_proc=='word':
                        new_txt.append(inst)

                    elif self.interj_proc=='drop':
                        continue

                    elif self.interj_proc=='token':
                        new_txt.append(self.interj_token)

                    else:
                        raise Exception(f'Invalid interjection process setting: {self.diff_proc}')

            else:
                new_txt.append(inst)

        return ' '.join(new_txt)


    def __call__(self, txt):
        txt = self.abbreviation_process(txt)
        txt = self.diffluency_process(txt)
        txt = self.inaudible_process(txt)
        txt = self.interjection_process(txt)
        txt = self.uncertain_process(txt)
        txt = self.numeric_process(txt)
        txt = txt.upper() if self.capitalize else txt.lower()
        txt = txt if self.apostrophe else txt.replace("'","")
        txt = txt if self.hypen else txt.replace("-"," ")
        txt = ' '.join(txt.split()).strip() # delete double space

        return txt


if __name__=="__main__":
    script_generator = Script_Generator(capitalize=True, 
                    apostrophe=False,
                    #numeric_format="pronunciation", 
                    numeric_format="digit", 
                    space_on_abbreviation=True, 
                    default_inaudible_process="token", 
                    default_uncertain_process="word", 
                    default_diffluency_process="word", 
                    default_interjection_process="word", 
                    interjection_to_word=['hmm'], 
                    interjection_to_token=['yeah'], 
                    interjection_to_drop=['mhm'], 
                    inaudible_token="<unknown>", 
                    uncertain_token="<unknown>", 
                    diffluency_token="<diffluency>", 
                    interjection_token="<interjection>" 
                    )
    #script_generator = Script_Generator()
    txt = "{hmm} !tv! blah blah (1.0/one point o) <th th> the [i don't know] (inaudible) blah blah {yeah} {mhm}"
    print(txt)
    print(script_generator(txt))
