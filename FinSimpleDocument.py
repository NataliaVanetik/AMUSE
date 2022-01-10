import os, sys, time, math, re
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
DOUBLE_SPACE = "  "
EOL="\n"
DOT="."
DEBUG=1


###################################################################
### Financial document and its data structures, neural & ordinary
###################################################################
class SimpleDocument:
    # create document by reading text from the file (default) or text(fromFile=False)
    def __init__(self, source, nlp, fromFile=True):
        self.DEBUG=1
        self.emptySentence=" "
        self.nlp=nlp
        if fromFile:
            self.readFromFile(source)
        else:
            self.readFromText(source)
                
    ######################################
    # read all the data from file and
    # parse it, including bert vectors
    ######################################
    def readFromFile(self, filename):
        # read from the file
        self.name=filename
        self.short_name=os.path.basename(filename).split(DOT)[0].lower()
        # leave sentenceNum sentences
        args = {'encoding': 'utf8', 'mode': 'rt'}
        with open(filename, **args) as file:
            self.text=file.read()
        if self.DEBUG>2:
            print("Read file=", filename)
        self.readFromText(self.text)
            
    ######################################
    # parse text data, including bert vectors
    ######################################        
    def readFromText(self, text):
        self.text=text
        if len(text)>1000000:
            self.text=self.text[:1000000]
        # process nlp basic
        self.doc = self.nlp(self.text)
        self.sentences=[]
       
        # regular pipeline
        sid=0
        
        for sentence in self.doc.sents:
            if sentence.text.strip()=="" or len(sentence.text.strip())<2:
                continue
            # sentence-level data
            if self.DEBUG>2:
                print("---> sid=",sid," is ",sentence.text)
            # save source sentence
            clean_text=sentence.text.replace(EOL,SPACE)
            clean_text=clean_text.replace(DOUBLE_SPACE,SPACE)
            clean_text=clean_text.strip()
            self.sentences.append(clean_text)
            
            sid=sid+1
            
        if self.DEBUG>0:
            print("orig doc ",self.name," has ", len(self.sentences), " sentences  ")
        self.sentenceNum=len(self.sentences)
        
    ######################################
    # check where scored sentences are
    # located within this document
    ######################################
    def matchScoredSentences(self, scored_sentences):
        self.score1_indexes=[]
        self.score0_indexes=[]
        self.unmatched_sentences=[]
        
        for key, scored in scored_sentences.items():
            sc_sent=scored['sentence']['sentence']
            if self.DEBUG>2:
                print(sc_sent)
            if sc_sent in self.sentences:
                ind=self.sentences.index(sc_sent)
                if self.DEBUG>2 and scored['score']==1:
                    print("   ---- sentence sid=",scored['sid']," found at index ", ind," with score ",scored['score'],"----")
                if scored['score']==1:
                    self.score1_indexes.append(ind)
                else:
                    self.score0_indexes.append(ind)
            else:
                ind=-1
                self.unmatched_sentences.append(sc_sent)
            if self.DEBUG>2 and sc_sent.find('chief operating and commercial officer')>=0:
                print("---------- found sentence in the test document -------- sid=",key," score=",scored['score']," ---------")
                print(sc_sent)
        if self.DEBUG>2:
            print("---------------- score 1 indexes in the original doc ---------------")
            print(self.score1_indexes)
        
    #################################################
    # count proper tokens in spacy sentence obj
    #################################################
    @staticmethod
    def token_count(sent):
        swc=0
        for token in sent:
            if re.search('[a-zA-Z]', token.text) is not None:
                swc=swc+1
        if DEBUG>2:
            print("       sentence of length=",swc," is ",sent.text)
        return swc
    
    #################################################
    # see if this is an empty or too short sentence
    #################################################
    @staticmethod
    def is_empty_sentence(sent):
        if DEBUG>2:
            print("       sent=",sent)
        if sent.text.strip()=="" or len(sent)<2:
            if DEBUG>2:
                print("       skipping sentence ",sent.text)
            return True
        return False
    
    #################################################
    # build summary from indexes with limit words
    #################################################
    def generate_summary(self, indexes, limit):
        # step 1 - select sentences with <limit> words
        limited_indexes=[]
        wc=0
        sid=0
        sents=list(self.doc.sents)
        for i in range(len(indexes)):
            sent=sents[indexes[i]]
        
            if SimpleDocument.is_empty_sentence(sent):
                continue
            # sentence-level data
            swc=SimpleDocument.token_count(sent)
            
            if wc+swc<=limit:
                # add sentence
                limited_indexes.append(indexes[i])
                # update wc
                wc=wc+swc
            
            if wc>=limit:
                break
            sid=sid+1
            
        if self.DEBUG>2:
            print("------------------ limited_indexes=",limited_indexes)
        # step 2 - sort the indexes
        limited_indexes.sort()
        if self.DEBUG>2:
            print("------------------ sorted limited_indexes=",limited_indexes)
        # step 3 - build final list of sentences
        summary=[]
        for i in limited_indexes:
            summary.append(sents[i].text+' ')
            
        return summary
    

    