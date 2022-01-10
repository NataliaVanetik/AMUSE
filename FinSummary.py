import os, sys, time, math
import random
import pulp
import itertools
import spacy
import en_core_web_sm  # basic EN model
import en_core_web_trf # basic transformer model (768)
from rouge import Rouge
import numpy as np
import random

SPACE = " "
EOL="\n"
UNDERSCORE="_"
SUMM="summ"
TXT_EXTENSION=".txt"

###################################################################
### Pattern and its instances in the database
###################################################################
class Summary:
    # create document by reading text from the file (default) or text(fromFile=False)
    def __init__(self, name, all_names, all_sentences, all_scores):
        self.DEBUG=1
        self.name=name
        self.summary=[]
        if self.DEBUG>0:
            print("Summary __init__: got ",name," names # ",len(all_names))
            if all_sentences is not None and all_scores is not None:
                print("     sentences # ",len(all_sentences)," scores # ",len(all_scores))
        # extract sentences of given document only in their original order
        self.sentences=dict()
        sid=0
        for i in range(len(all_names)):
            if all_names[i]==name:
                self.sentences[sid]=dict()
                self.sentences[sid]['sid']=sid
                self.sentences[sid]['score']=all_scores[i]
                self.sentences[sid]['sentence']=all_sentences[i] # contains all sentence info from Document
                sid=sid+1
        if self.DEBUG>0:
            print("     document ",name," extracted ",sid," sentences with scores")
            
    #####################################################################
    # build the summary: select sentences according to scores,
    # make sure the size does not exceed maxwords, and then arrange them
    # by sid
    #####################################################################
    def build(self, maxwords):
        if self.DEBUG>2:
            print("self.sentences.values()[0]=",list(self.sentences.values())[0])
        # lambda x: (x[1], x[2])
        sorted_sentences=sorted(list(self.sentences.values()), key=lambda item: (1-item['score'], item['sid']))
        if self.DEBUG>2:
            print("sorted by score,sid=",[s['sid'] for s in sorted_sentences])
            print("    scores         =",[s['score'] for s in sorted_sentences])
            print("    wc             =",[s['sentence']['wc'] for s in sorted_sentences])
            j=0
            for s in sorted_sentences:
                if s['sentence']['wc']<=5:
                    print("\t\t wc=",s['sentence']['wc']," sentence #",j,"=",s['sentence']['sentence'])
                j=j+1
        if self.DEBUG>2:
            print("    maxwords       =",maxwords)
        wc=0
        i=0
        selected_sentences=[]
        for sent in sorted_sentences:
            if self.DEBUG>2:
                print("sorted_sentence[",i,"=",sent)
            if sent['sentence']['wc']<5:
                continue
            if self.DEBUG>3 and sent['sentence']['wc']<7:
                print("Short sentence detected: ",sent['sentence']['sentence'])
            if wc+sent['sentence']['wc']<=maxwords: # skip too-short sentences
                selected_sentences.append(sent)
                wc=wc+sent['sentence']['wc']
            else:
                break
            i=i+1
        if self.DEBUG>2:
            print("selected          =",[s['sid'] for s in selected_sentences])
            print("    scores        =",[s['score'] for s in selected_sentences])
            print("    wc            =",[s['sentence']['wc'] for s in selected_sentences])
            print("    total wc      =",wc)
            
        selected_sorted_sentences=sorted(selected_sentences, key=lambda item: item['sid'])
        if self.DEBUG>2:
            print("selected  sorted  =",[s['sid'] for s in selected_sorted_sentences])
            print("    scores        =",[s['score'] for s in selected_sorted_sentences])
            print("    wc            =",[s['sentence']['wc'] for s in selected_sorted_sentences])
        
        self.summary=[key['sentence']['sentence'] for key in selected_sorted_sentences]
        self.summaryText=""
        for s in self.summary:
            # check for duplicates
            if s in self.summaryText and self.DEBUG>0:
                print("\t\t found duplicate sentence =",s)
            self.summaryText=self.summaryText+s+' '
        if self.DEBUG>2:
            print("Constructed summary for document=",self.name)
            print("Summary=",self.summary)
    
    #####################################################################    
    # store summary in its dedicated file within given directory
    #####################################################################
    def store(self, summdir):
        if not os.path.exists(summdir):
            os.makedirs(summdir)
        filename=self.name+UNDERSCORE+SUMM+TXT_EXTENSION
        args = {'encoding': 'utf8', 'mode': 'w'}
        fname=os.path.join(os.getcwd(),summdir,filename)
        if self.DEBUG>0:
            print("Storing summary for document=",self.name," in file=",fname)
        with open(fname, **args) as file:
            for sent in self.summary:
                file.write(sent+' ')