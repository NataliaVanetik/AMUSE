import networkx as nx
import argparse
from argparse import ArgumentParser
import os, sys, time, csv, pickle
import random
import pulp
import itertools
import pandas as pd
import spacy
import en_core_web_sm
import en_core_web_trf
import spicy
import scipy.stats as stats
from rouge import Rouge
from FinDocument import Document
from FinSimpleDocument import SimpleDocument
import numpy as np
from FinModel import Model
from FinSummary import Summary
from spacytextblob.spacytextblob import SpacyTextBlob
from rouge_metric import PyRouge
from tensorflow import keras
from imblearn.under_sampling import NearMiss
import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy
import nltk

#nlp = en_core_web_sm.load()
#doc = nlp("This is a sentence.")

DOT="."
UNDERSCORE="_"

####################################################################
# create list of prefixes for files in the list
####################################################################
def get_prefix_list(files):
    plist=[]
    for f in files:
        prefix=os.path.basename(f).split(DOT)[0].lower()
        if UNDERSCORE in prefix:
            prefix=prefix.split(UNDERSCORE)[0].lower()
        plist.append(prefix)
    return plist

####################################################################
# Read one document, create its data structures, and find
# matching gold summaries & process them too
####################################################################
def processOneDocument(textFile, goldfiles, gold_prefixes, nlp, nlpT, fullProcessing):
    # read the text file
    textDoc = Document(textFile, nlp, nlpT, True, fullProcessing)
    textPrefix=os.path.basename(textFile).split(DOT)[0].lower()
    
    if UNDERSCORE in textPrefix:
        textPrefix=textPrefix.split(UNDERSCORE)[0].lower()
    #print("========================> Text file prefix is ",textPrefix)
    
    goldDocs=[]
    # go over gold files, find matchings
    if goldfiles is not None and gold_prefixes is not None:
        maxGoldSentences=0
        
        indices = [i for i, x in enumerate(gold_prefixes) if x == textPrefix]
        gf=[goldfiles[i] for i in indices]
        if DEBUG>2:
            print("============ textPrefix          ",textPrefix)
            print("             gold indices        ",indices)
            print("             matching gold files ",gf)
        if goldfiles is not None and len(gf)>0:
            for goldFile in gf:
                goldDoc = Document(goldFile, nlp, nlpT)
                goldDocs.append(goldDoc)
            if DEBUG>0:
                print("========================> Text file prefix is ",textPrefix," found ",len(goldDocs)," matching gold summaries")

    if DEBUG>2:
        print("========================> Data build done ")
    return textDoc, goldDocs


####################################################################
# Create summaries from text predictions
####################################################################
def createSummaries(doc_names, sentences, scores, args, docs, nlp):
    exists = os.path.isdir(args.summarydir)
    if not exists:
        print("Summary directory ",args.summarydir," does not exist, creating")
        os.makedirs(args.summarydir)
        
    # generate summaries
    doc_list=list(set(doc_names))
    for i in range(len(doc_list)):
        dname=doc_list[i]
        summary=Summary(dname, doc_names, sentences, scores)
        summary.build(args.sumlen)
        summary.store(args.summarydir)
        if DEBUG>0:
            print("========== stored summaries for file ",i,"/",len(doc_list))
        # find the doc for this summary
        doc=None
        for d in docs:
            if d.short_name==dname:
                doc=d
                break
        if doc is not None:
            print("=========== Setting summary for doc ",doc.short_name)
            doc.setSummary(summary)
            
##############################################################################
# define program arguments
##############################################################################
def define_parser():
    parser = argparse.ArgumentParser(description="Run fin summarization")
    parser.add_argument('--train', type=str, default='data/train/', help='train text dir')
    parser.add_argument('--test', type=str, default='data/test/', help='test text dir')
    parser.add_argument('--count', type=int, default=1, help='max # of files to process, -1 for all')
    parser.add_argument('--start', type=int, default=0, help='# of file to start from')
    parser.add_argument('--gold', type=str, default='data/gold/', help='GS summary dir')
    parser.add_argument('--task', type=str, default='train', help='task type', \
                        choices=['train','generate-summaries'])
    parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
    parser.add_argument('--sumlen', type=int, default=1000, help='size of a summary (words)')
    parser.add_argument('--summarydir', type=str, default='data/summaries/', help='directory to store generated summaries')
    parser.add_argument('--modelpath', type=str, default='data/models/model.h5', help='file to store trained model')
    return parser

##############################################################################
# check existence of a directory
##############################################################################
def check_directory(dirname):
    direct = os.path.join(os.getcwd(), dirname)
    exists = os.path.isdir(direct)
    if not exists:
        print("Directory ",direct," does not exist!!!")
        sys.exit(0)
    return direct

##############################################################################
# read file names from directories
##############################################################################
def read_file_names(trainDir, goldDir, testDir):
    # read train directory, get text files
    trainfiles = [os.path.join(trainDir, f) for f in os.listdir(trainDir) if os.path.isfile(os.path.join(trainDir, f))]
    print("Found ",len(trainfiles)," files in the text directory ", trainDir)
       
    # read gold directory, get matching gold summaries
    goldfiles = [os.path.join(goldDir, f) for f in os.listdir(goldDir) if os.path.isfile(os.path.join(goldDir, f))]
    print("Found ",len(goldfiles)," files in the gold directory ", goldDir)
    
    # read test directory, get text files
    testfiles=[]
    if os.path.isdir(testDir):
        testfiles = [os.path.join(testDir, f) for f in os.listdir(testDir) if os.path.isfile(os.path.join(testDir, f))]
        print("Found ",len(trainfiles)," files in the text directory ", testDir)
    return trainfiles, goldfiles, testfiles


######################################################################
# generate training documents ONLY
######################################################################
def generate_train_data(trainfiles, goldfiles, args, nlp, nlpT, fullProcessing):
    # process train and test data, get max sentence number
    # limit processing to #count files for both train and test
    gold_prefixes=get_prefix_list(goldfiles)

    # process training data
    start=0
    if args.start>0:
        start=args.start
    end=len(trainfiles)
    if args.count>0:
        end=min(start+args.count, len(trainfiles))
    if DEBUG>0:    
        print("--------------------------- loading files #",start,"..",end)    

    trainDocs=[]
    trainGolds=[]
    i=0
    for i in range(start,end):
        textFile=trainfiles[i]
        print("Processing file #",i," name=",textFile)
        doc, golds = processOneDocument(textFile, goldfiles, gold_prefixes, nlp, nlpT, fullProcessing)
        trainDocs.append(doc)
        trainGolds.append(golds)
        i=i+1
        
    return  trainDocs, trainGolds

######################################################################
# generate test documents ONLY
######################################################################
def generate_test_data(testfiles, args, nlp, nlpT, fullProcessing):
    # process test data, get max sentence number
    # limit processing to #count files 
    # process training data
    start=0
    if args.start>=0:
        start=args.start
    end=len(testfiles)
    if args.count>0:
        end=min(start+args.count, len(testfiles))
    if DEBUG>0:    
        print("--------------------------- testing on files #",start,"..",end)    

    testDocs=[]
    i=0
    for i in range(start,end):
        textFile=testfiles[i]
        print("Processing file #",i," name=",textFile)
        doc, golds = processOneDocument(textFile, None, None, nlp, nlpT, fullProcessing)
        testDocs.append(doc)
        i=i+1
    return  testDocs

######################################################################
# get file names, shortened
######################################################################
def get_short_file_names_list(testfiles, args):
    # process test data, get max sentence number
    # limit processing to #count files 
    # process training data
    start=0
    if args.start>=0:
        start=args.start
    end=len(testfiles)
    if args.count>0:
        end=min(start+args.count, len(testfiles))
    if DEBUG>0:    
        print("--------------------------- processing files #",start,"..",end)    

    names=[]
    i=0
    for i in range(start,end):
        fname=testfiles[i]
        if DEBUG>0:
            print("Got file #",i," name=",fname)
        shortName=os.path.basename(fname).split(DOT)[0].lower()
        names.append(shortName)
        i=i+1
    return  names


######################################################################
# generate training data only for NN
######################################################################
def generate_train_neural_data(trainDocs, trainGolds, args):
    train_X=[]
    train_Y=[]
    # process training data
    for i in range(len(trainDocs)):
        doc=trainDocs[i]
        golds=trainGolds[i]
        doc.computeNodeVectors()
        doc.computeSentenceLabels(golds)
        if doc.empty_data==False: # there is a data
            ddata_X=doc.getSentenceData()
            if DEBUG>0:
                print("Got train doc ",i,"/",len(trainDocs),"  ",doc.short_name," data of shape ",np.asarray(ddata_X).shape)
            train_X.extend(ddata_X)
            
            ddata_Y=None
            ddata_Y=doc.getBinarySentenceLabels()
            train_Y.extend(ddata_Y)
            if DEBUG>0:
                print("Got binary labels ",ddata_Y)
        else:
            if DEBUG>0:
                print("Got train doc ",i,"/",len(trainDocs),"  with empty data, skipping ")
        
    if DEBUG>0:    
        print("---------------------------------------------------------------")
        print("Training data size = ",len(train_X)," shape=",np.asarray(train_X).shape," labels shape=",np.asarray(train_Y).shape)
        print("---------------------------------------------------------------")
        
    return  train_X, train_Y

######################################################################
# generate test data only for NN
######################################################################
def generate_test_neural_data(testDocs, args):
    test_X=[]
    test_sentences=[]
    test_doc_names=[]
    
    # process training data
    for i in range(len(testDocs)):
        doc=testDocs[i]
        doc.computeNodeVectors()
        ddata_X=doc.getSentenceData()
        if DEBUG>0:
            print("Got test doc ",i,"/",len(testDocs),"  ",doc.short_name," data of shape ",np.asarray(ddata_X).shape)
        test_X.extend(ddata_X)
        if DEBUG>2:
            print("Got test doc ",i,"/",len(testDocs),"  ",doc.short_name," data ")
        for i in range(len(doc.sentences)):
            test_doc_names.append(doc.short_name)
            test_sentences.append(doc.sentences[i])
        
    if DEBUG>0:    
        print("---------------------------------------------------------------")
        print("Test data size = ",len(test_X)," shape=",np.asarray(test_X).shape)
        print("---------------------------------------------------------------")
        
    return  test_X, test_doc_names, test_sentences

    
##############################################################################
# read file names from directories
##############################################################################
def read_file_names(direct):
    # read train directory, get text files
    files = [os.path.join(direct, f) for f in os.listdir(direct) if os.path.isfile(os.path.join(direct, f))]
    print("Found ",len(files)," files in directory ", direct)
    return files


##########################################################################################################
# MAIN PART
##########################################################################################################
DEBUG=1
parser = define_parser()
args = parser.parse_args()

nlp = en_core_web_sm.load()
nlp.add_pipe('spacytextblob')
nlpT = en_core_web_trf.load()

nltk.download('punkt')
if DEBUG>2:
    print("------------> nlp.pipe_names =",nlp.pipe_names)
# if the task is 'evaluate', collect all data,do train/test split, run & compute accuracy/value
# oftherwise, train on train_X and produce predictions for test_X


################################################################################
print("---------------- Task == ",args.task," -------------------")
model=None

if args.task=='train': 
    # train and save a generated model
    trainDir = check_directory(args.train)
    goldDir = check_directory(args.gold)
    trainfiles = read_file_names(trainDir)
    goldfiles = read_file_names(goldDir)
    # generate data
    # generate train and test data documents
    trainDocs, trainGolds=generate_train_data(trainfiles, goldfiles, args, nlp, nlpT, fullProcessing=True)
    # generate data for NN
    train_X, train_Y = generate_train_neural_data(trainDocs, trainGolds, args)
    # train on the whole train data, predict on test data
    model=Model(train_X, train_Y, args.epochs, None, None, True)
    #model.predict
    model.compile
    model.fit
    # save model in file
    #with open(args.modelpath, 'wb') as mfile:
    #    pickle.dump(model, mfile)
    model.model.save(args.modelpath)
elif args.task=='generate-summaries':
    # load model, generate data and produce summaries with this model
    # check that a file exists
    exists = os.path.isfile(os.path.join(os.getcwd(), args.modelpath))
    if not exists:
        print("Model file ",args.modelpath," does not exist!!!")
        sys.exit(0)   
    # load model, do more training, save model
    trained_model = keras.models.load_model(args.modelpath)
    if DEBUG>0:
        print("Loaded model from file ",args.modelpath)
    # set up test data
    testDir = check_directory(args.test)
    testfiles = read_file_names(testDir)
    
    if DEBUG>0:
        print("Loaded test files from ",args.test)
    # generate data
    # generate train and test data documents
    testDocs=generate_test_data(testfiles, args, nlp, nlpT, fullProcessing=True)
    # generate data for NN
    test_X, test_doc_names, test_sentences = generate_test_neural_data(testDocs, args)
    model=Model(None, None, args.epochs, test_X, None, True, trained_model)
    predictions = model.get_predictions()
    # create summaries from predictions
    createSummaries(test_doc_names, test_sentences, predictions, args, testDocs, nlp)
else:
    print("!!!   Unsupported task type ",args.task," !!!")


