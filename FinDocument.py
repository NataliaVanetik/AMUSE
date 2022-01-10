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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cdist
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from statistics import median
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.centrality import girvan_newman
import community as community_louvain
import distance
from nltk import Tree

SPACE = " "
EOL="\n"
DOT="."
MASK="MMMMASK"
DEBUG=2
NODE2VEC_DIMS=128 #128
ITER=8 # 1
NUM_WALKS=10 #10
WALK_LENGTH=80

###################################################################
### Financial document and its data structures, neural & ordinary
###################################################################
class Document:
    # create document by reading text from the file (default) or text(fromFile=False)
    def __init__(self, source, nlp, nlpTransformers, fromFile=True, fullProcessing=True):
        self.CREATE_TOKEN_MATRICES=fullProcessing
        self.DEBUG=1
        self.emptySentence=" "
        self.nlp=nlp
        self.nlpT = nlpTransformers
        self.sentenceMatrix=None
        self.tokenVectorMatrix=None
        self.empty_data=False
        if fromFile:
            self.readFromFile(source)
        else:
            self.readFromText(source)
        self.clusters=None
        self.summary=None
        
                
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
        # process nlp basic
        self.doc = self.nlp(self.text)
        self.sentences=[]
        self.unique_tokens=set()
        self.__node2vec_dimensions=NODE2VEC_DIMS
        self.Rouge=Rouge()
       
        # regular pipeline
        sid=0
        
        for sentence in self.doc.sents:
            if sentence.text.strip()=="" or len(sentence.text.strip())<2:
                continue
            # sentence-level data
            sdict=dict()
            if self.DEBUG>2:
                print("---> doc ",self.short_name," sid=",sid," is ",sentence.text.strip())
            # save source sentence
            sdict['sentence']=sentence.text.strip()
            sdict['sid']=sid
            sdict['node-embedding']=dict()
            sdict['tree']=Document.to_nltk_tree(sentence.root)
            
            if self.DEBUG>2:
                print("sid=",(sid-1)," tree=",sdict['tree'])
                
            sdict['nx-tree']=Document.nx_tree(sid, sentence.root)
            root_e=Document.root_dependency_embedding(sid, sentence.root, sdict['nx-tree'])
            sdict['dep-tree-root']=[root_e, root_e] # TBD: replace it by better data
            
            if self.DEBUG>2:
                print("sid=",(sid-1)," nx-tree=",list(sdict['nx-tree'].nodes(data=True)))
            
            sid=sid+1
            
            if self.DEBUG>2:
                print("sid=",(sid-1)," polarity=",sentence._.polarity)     # Polarity: -0.125
                print("sid=",(sid-1),"subjectivity=", sentence._.subjectivity)  # Sujectivity: 0.9
                print("sid=",(sid-1),"assessments=", sentence._.assessments)
            sent_sentiment_vec=[sentence._.polarity,sentence._.subjectivity]
            sdict['sentiment']=sent_sentiment_vec #token_sentiment_vec
            
            swc=0
            for token in sentence:
                if re.search('[a-zA-Z]', token.text) is not None:
                    swc=swc+1
            sdict['wc']=swc  #len(tokens)
            
            if self.CREATE_TOKEN_MATRICES:
                # token-level data
                tokens=[]
                sum_token_vec=[]
                ntok=0
                token_text_vec=[]
                token_pos_vec=[]
                token_lemma_vec=[]
                token_tag_vec=[]
                token_dep_vec=[]
                token_sentiment_vec=[]
                
                for token in sentence:
                    if self.DEBUG>2:
                        print("Token data=",token.text, token.pos_, token.dep_, token.lemma_, token.tag_, \
                              token.has_vector, token.vector_norm, token.is_oov, token.vector)
                    if self.DEBUG>2 and token.sentiment!=0:
                        print("Token=",token)
                        print("Token sentiment data=",token.sentiment)
                    if self.DEBUG>2:
                        print("Token=",token)
                        print("Dependency=",token.dep_)
                    tokens.append(token)
                    self.unique_tokens.add(token)
                    token_text_vec.append(token.text.lower())
                    token_lemma_vec.append(token.lemma_)
                    token_pos_vec.append(token.pos_)
                    token_tag_vec.append(token.tag_)
                    token_dep_vec.append(token.dep_)
                    token_sentiment_vec.append(token.sentiment)
                
                    if self.DEBUG>2:
                        print("---------------> sid=",(sid-1)," tid=",ntok," is ",token.text)
                    self.token_vector_size=len(token.vector)
                    if ntok==0:
                        sum_token_vec=token.vector
                    else:
                        for i in range(len(token.vector)):
                            sum_token_vec[i] = sum_token_vec[i]+token.vector[i]
                    ntok=ntok+1
                
                for i in range(len(sum_token_vec)):
                    sum_token_vec[i] =sum_token_vec[i]/ntok
                    
                if self.DEBUG>1:
                    print("---------------> sid=",(sid-1)," token_text_vec size=",len(token_text_vec))
            
                
                for i in range(len(token.vector)):
                    sum_token_vec[i] = sum_token_vec[i]/ntok
                sdict['tokens']=tokens
                '''
                swc=0
                for token in tokens:
                    if re.search('[a-zA-Z]', token.text) is not None:
                        swc=swc+1
                sdict['wc']=swc  #len(tokens)
                '''
                sdict['average-WE']=np.array(sum_token_vec)
                sdict['tags']=token_tag_vec
                sdict['pos']=token_pos_vec
                sdict['dep']=token_dep_vec
               
                if self.DEBUG>1:
                    print("Sentence ",(sid-1)," token vector has shape",sdict['average-token-embedding'].shape)
            
            # entities
            entities=[]
            for ent in sentence.ents:
                if self.DEBUG>2:
                    print(ent.text, ent.start_char, ent.end_char, ent.label_)
                entities.append(ent.text)
            sdict['ner']=entities
            
            # transformer pipeline
            tdoc=self.nlpT(sentence.text)
            tokvec=tdoc._.trf_data.tensors[-1]
            if self.DEBUG>1:
                print("tokvecs.shape=", (np.array(tokvec)).shape, " tokvec[0].shape=",(np.array(tokvec[0])).shape)
            sdict['sentence-embedding']=np.array(tokvec)
            sdict['sentence-embedding-sum']=sum(tokvec[0])
            self.sentence_vector_size=len(tokvec)
            
            self.sentences.append(sdict)
        
        if self.DEBUG>2:
            print("doc ",self.name," has ", len(self.sentences), " sentences and ",len(self.unique_tokens)," unique tokens")
        self.sentenceNum=len(self.sentences)
        
        # sizes
        self.num_of_sentences=len(self.sentences)
        self.num_of_tokens=len(list(self.unique_tokens))
        
    ######################################################################
    # Parse results to nltk tree
    ######################################################################
    @staticmethod
    def to_nltk_tree(node):
     if node.n_lefts + node.n_rights > 0:
         return Tree(node.orth_, [Document.to_nltk_tree(child) for child in node.children])
     else:
         return node.orth_
        
    ######################################################################
    # Dependency nodes as a list
    ######################################################################
    @staticmethod
    def get_dep_nodes(node):
        #print("============> node=",node," type=",type(node))
        res=[node.i] # was: [node.orth_] replaced due to possible duplicates
        '''
        if node.orth_ is None or node.orth_.strip()=='':
            return []
        '''
        if node.n_lefts + node.n_rights > 0:
           for child in node.children:
               res.extend(Document.get_dep_nodes(child))
           return res
        else:
           return res
        
    ######################################################################
    # Dependency node parameters as a list: Wv of a word+ int dependency
    ######################################################################
    @staticmethod
    def get_dep_nodes_parameters(node):
        #print("============> node=",node.orth_," dependency=",node.dep_," int dep=",node.dep,\
        #      " has vector=", node.has_vector, " vector=", node.vector)
        #print(list(node.vector))
        
        '''
        if node.orth_ is None or node.orth_.strip()=='':
            if DEBUG>3:
                print("    returning empty node param vector for node ", node.orth_, "#",node.i)
            return [[]]
        '''
        res=[node.dep]
        res.extend(list(node.vector))
        res=[res]
        if DEBUG>3 and node.orth_.strip()=='last':
            print("-------------------------- node=",node.orth_," initial vector=",res)
        
        if node.n_lefts + node.n_rights > 0:
            for child in node.children:
                res.extend(Document.get_dep_nodes_parameters(child))
            if DEBUG>3 and node.orth_.strip()=='last':
                print("-------------------------- node=",node.orth_," updated vector=",res)
            return res
        
        if DEBUG>3 and node.orth_.strip()=='last':
            print("-------------------------- node=",node.orth_," final vector=",res)
        return res
    
    ######################################################################
    # Dependency edges as a list
    ######################################################################
    @staticmethod
    def get_dep_edges(node):
        if node.n_lefts + node.n_rights > 0:
           res=[]
           for child in node.children:
                #if child is not None and child.orth_.strip()!='':
                res.append((node.i, child.i))
                res.extend(Document.get_dep_edges(child))
           return res
        else:
           return []
      
    ######################################################################
    # Parse results to nx tree
    ######################################################################  
    @staticmethod    
    def nx_tree(sid, node):
        """Convert the data in a ``nodelist`` into a networkx labeled directed graph."""
        import networkx
        
        nx_nodelist = Document.get_dep_nodes(node)
        nx_edgelist = Document.get_dep_edges(node)
        nx_nodeparams = Document.get_dep_nodes_parameters(node)
        
        #print("Nodes=",nx_nodelist)
        #print("Edges=",nx_edgelist)
        #print("Node params=",nx_nodeparams)
        #print("Got node params of length=",len(nx_nodeparams)," for ",len(nx_nodelist)," nodes")
        
        g = networkx.DiGraph()
        for i in range(len(nx_nodelist)):
            g.add_node(nx_nodelist[i],vector=nx_nodeparams[i])
            if DEBUG>2 and len(nx_nodeparams[i])<=1: 
                print("node #",i,"/",len(nx_nodelist)," node=[",nx_nodelist[i].strip(),\
                      "] vector=",nx_nodeparams[i]," total node params =",len(nx_nodeparams))

        g.add_edges_from(nx_edgelist)
        
        #print("Graph=",g)
        if DEBUG>2:
            print("Sentence #",sid," has ",len(nx_nodelist)," nodes, its graph has ",len(list(g.nodes()))," nodes")
        return g
        
    ######################################################################
    # Generate topK summary
    ######################################################################
    def get_topK_summary(self, k):
        topK=""
        wc=0
        i=0
        
        for i in range(len(self.sentences)):
            if self.DEBUG>2:
                print("adding sentence #",i)
            if wc+self.sentences[i]['wc']<=k:
                topK=topK+self.sentences[i]['sentence']+' '
                wc=wc+self.sentences[i]['wc']
            else:
                break
            i=i+1
        return topK
       
    ######################################################################
    # Matrices to VSM matrices in common space
    ######################################################################
    @staticmethod
    def transformMatrices(matrix1, matrix2):
        # first, create dictionary
        mset=set()
        for row in matrix1:
           mset.update(row)
        for row in matrix2:
           mset.update(row)
        mlist=list(mset)
        
        if DEBUG>2:
            print("matrix element list of size ",len(mlist),"=",mlist)
            
        msize=len(mlist)
        
        matrix1new=[]
        for row in matrix1:
            row1new=[0] * msize
            for element in row:
                row1new[mlist.index(element)]=row1new[mlist.index(element)]+1
            matrix1new.append(row1new)
        
        matrix2new=[]
        for row in matrix2:
            row2new=[0] * msize
            for element in row:
                row2new[mlist.index(element)]=row2new[mlist.index(element)]+1
            matrix2new.append(row2new)
        if DEBUG>2:
            print("got new matrix ",matrix2new)
            
        return matrix1new, matrix2new
       
    '''
    ###########################################################   
    # get sentence dict by id
    ###########################################################
    def getSentenceData(self, sid):
        return self.sentences[sid]
    '''
    
    ##############################################
    # compute representation matrices
    ##############################################
    def getSentenceMatrix(self):
        if self.sentenceMatrix is None:
            # compute
            self.sentenceMatrix = np.concatenate([sdict['sentence-embedding'] for sdict in self.sentences], axis=0)
        if self.DEBUG>1:
            print("Generated sentence matix of shape ",self.sentenceMatrix.shape)
        return self.sentenceMatrix
    
    def getTokenVectorMatrix(self):
        if self.tokenVectorMatrix is None:
            # compute
            tlist=[sdict['average-token-embedding'] for sdict in self.sentences]
            #print(np.asarray(tlist).shape)
            self.tokenVectorMatrix = np.array(tlist)
        if self.DEBUG>1:
            print("Generated token matix of shape ",self.tokenVectorMatrix.shape)
        return self.tokenVectorMatrix
    
    ################################################################
    # node embeddings
    ################################################################
    # run node embeddings algorithm
    @staticmethod
    def learn_embeddings(graph, walks):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [list(map(str, walk)) for walk in walks]
        
        if DEBUG>2:
            print("======> Walk mapping done ")
        if DEBUG>2:
            for i in range(min(3,len(walks))):
                print("        mapped walk[",i,"]=",walks[i])
           
        # default parameter values        
        WINDOW_SIZE=10
        WORKERS = 8
        ITER=1
        model = Word2Vec(walks, vector_size=self.__node2vec_dimensions, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS)#, iter=ITER)
        if DEBUG>2:
            print("======> Node2vec model constructed")
        #model.wv.save_word2vec_format(args.output) TBD - save later
        return model
    
    ################################################################
    # compute node embeddings based selected data
    ################################################################
    def generateNodeEmbeddings(self, dname):
        ###################### build directed graph with token nodes and dependency nodes ###########
        sentence_embedding_graph = self.get_sentence_graph(data_name=dname, prune=True, inverseDistances=True) # was: True
        if sentence_embedding_graph.number_of_nodes()<=0:
            self.empty_data=True
            return None, None
        ############ hyperparameters and graph type ###############
        P=1
        Q=1
        DIRECTED = 'directed'
        G = node2vec.Graph(sentence_embedding_graph, DIRECTED, P, Q)
        G.preprocess_transition_probs()
        ############ random walk parameters, meanwhile set to default
        #NUM_WALKS=10
        #WALK_LENGTH=80
        walks = G.simulate_walks(NUM_WALKS, WALK_LENGTH)   
        model = Document.learn_embeddings(sentence_embedding_graph, walks)
        if self.DEBUG>2:
            print("=========> Got ",len( model.wv.vectors)," vectors of shape=",\
                  model.wv.vectors.shape," for ",len(sentence_embedding_graph.nodes)," nodes")
            
        # store node embeddings
        i=0
        for n in sentence_embedding_graph.nodes:
            if self.DEBUG>2:
                print(dname," embedding of node=",n,"of ",len(sentence_embedding_graph.nodes),\
                      " is vector of length ",len(model.wv.vectors[i]))
            if self.DEBUG>2:
                print("\t =",model.wv.vectors[i])
            self.sentences[i]['node-embedding'][dname]=model.wv.vectors[i]
            i=i+1
        
        return model, sentence_embedding_graph
    
    ###############################################################
    # Generate node embeddings for this tree (nx.DiGraph with bert
    # SE of words at nodes) and extract the root embedding
    ###############################################################
    def root_dependency_embedding(sid, root, graph):
        if DEBUG>2:
            print("root_dependency_embedding root  =",root," root ID=",root.i)
            print("root_dependency_embedding Nodes =",list(graph.nodes))
            print("root_dependency_embedding Edges =",list(graph.edges))
            
        ###################### set edge weights ###################
        for u,v,d in graph.edges(data=True):
            d['weight']=1
            if DEBUG>2:
                print("root_dependency_embedding edge data set=",graph[u][v])
        
        ############ hyperparameters and graph type ###############
        P=1
        Q=1
        DIRECTED = 'directed'
        G = node2vec.Graph(graph, DIRECTED, P, Q)
        G.preprocess_transition_probs()
        ############ random walk parameters, meanwhile set to default
        #NUM_WALKS=10
        #WALK_LENGTH=80
        walks = G.simulate_walks(NUM_WALKS, WALK_LENGTH)   
        model = Document.learn_embeddings(graph, walks)
        if DEBUG>2:
            print("=========> Got ",len(model.wv.vectors)," vectors of shape=",\
                  model.wv.vectors.shape," for ",len(graph.nodes)," nodes")
            
        # extract node embeddings
        i=0
        re=None
        for n in graph.nodes:
            if DEBUG>2:
                print(" embedding of node=",n,"of ",len(graph.nodes),\
                      " is vector of length ",len(model.wv.vectors[i]))
                print("\t =",model.wv.vectors[i])
            if n==root.i:
                re=model.wv.vectors[i]
                if DEBUG>2:
                    print("root_dependency_embedding RE set to ",re)
            i=i+1
        return re
    
    ###############################################################
    # alternative jaccard
    ###############################################################
    @staticmethod
    def alternative_jaccard_distance(list1, list2):
        if len(list1)+len(list2)<=0:
            return 1
        s1 = set(list1)
        s2 = set(list2)
        return 1-float(len(s1.intersection(s2)) / len(s1.union(s2)))
    
    ###############################################################
    @staticmethod
    def needsJaccard(name):
        return name in ['pos','dep','ner','tags','sentiment']
                
    ###############################################################
    # BERT sentence graph
    ###############################################################
    def get_sentence_graph(self, data_name='sentence-embedding', prune=True, inverseDistances=True):
        graph = nx.DiGraph()
            
        # simarray for median
        simarr=[]
        # generate edges between sentences
        for i in range(0, self.num_of_sentences):
            for j in range(i+1, self.num_of_sentences):
                graph.add_edge(i, j)
                graph.add_edge(j, i)
                # get both data vectors
                if self.DEBUG>2 and i==0 and j==1 and data_name=='sentiment':
                    print("self.sentences[",i,"][",data_name,"]=",self.sentences[i][data_name])
                    print("self.sentences[",j,"][",data_name,"]=",self.sentences[j][data_name])
                sim=1
                if Document.needsJaccard(data_name):
                    if len(self.sentences[i][data_name])+len(self.sentences[j][data_name])>0:
                        sim = distance.jaccard(set(self.sentences[i][data_name]),set(self.sentences[j][data_name]))
                        if not inverseDistances:
                            sim=1-sim
                        #sim1=Document.alternative_jaccard_distance(self.sentences[i][data_name],self.sentences[j][data_name])
                        if self.DEBUG>2 and i==0 and j==1 and data_name=='sentiment':
                            print("self.sentences[",i,"][",data_name,"]=",self.sentences[i][data_name])
                            print("self.sentences[",j,"][",data_name,"]=",self.sentences[j][data_name])
                            print("Found jaccard distance ",i,"<->",j,"=",sim)
                else:
                    sim = cosine_similarity(self.sentences[i][data_name],self.sentences[j][data_name])
                    if inverseDistances:
                        sim=1-sim[0][0]
                bert_distance=1-cosine_similarity(self.sentences[i]['sentence-embedding'],self.sentences[j]['sentence-embedding'])[0][0]        
                if self.DEBUG>3 and i==0 and j==1:
                    print("Found distance ",i,"<->",j,"=",sim)
                graph[i][j]['weight'] = sim
                graph[j][i]['weight'] = sim
                simarr.append(sim)
             
        if self.DEBUG>2:
            print("simarr=",simarr)
        med=0
        if simarr is not None and len(simarr)>0:
            med = median(simarr)
        if prune:
            pc=0
            for i in range(0, self.num_of_sentences):
                for j in range(i+1, self.num_of_sentences):
                    if graph[i][j]['weight']<med:
                        graph.remove_edge(i,j)
                        graph.remove_edge(j,i)
                        pc=pc+1
            if self.DEBUG>2:
                print("Pruned out total of ",pc," edges")
        
            
        #graph = graph.to_undirected()
        if self.DEBUG>2:
            print("=== ",data_name," ===> Created graph with ",graph.number_of_nodes()," nodes and ",graph.number_of_edges()," edges")
            
        if self.DEBUG>2:
            print("=== ",data_name," ===> Median edge weight is ",med)
            print("Nodes =",list(graph.nodes))
            print("Edges =",list(graph.edges))
            print(graph.edges.data())
        return graph
    
    ###############################################################
    # run node embeddings algorithm
    ###############################################################
    @staticmethod
    def learn_embeddings(graph, walks):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [list(map(str, walk)) for walk in walks]
        
        if DEBUG>2:
            print("======> Walk mapping done ")
        if DEBUG>2:
            for i in range(min(3,len(walks))):
                print("        mapped walk[",i,"]=",walks[i])
           
        # default parameter values        
        WINDOW_SIZE=10
        WORKERS = 8
        #ITER=1
        #NODE2VEC_DIMS=128
        model = Word2Vec(walks, vector_size=NODE2VEC_DIMS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS)#, iter=ITER)
        if DEBUG>2:
            print("======> Node2vec model constructed")
        #model.wv.save_word2vec_format(args.output) TBD - save later
        return model
    
    ###############################################################
    # get R1 & R2 scores for self.summary 
    ###############################################################
    def getRougeScoresFor(self, golds, text):
        scores=[]
        for g in golds:
            s = self.computeRougeForText(g, text)
            scores.append(s[0])
        # select the best (or avg)
        best=dict()
        best['rouge-1']=dict()
        best['rouge-2']=dict()
        best['rouge-1']['f']=max([s['rouge-1']['f'] for s in scores])
        best['rouge-1']['r']=max([s['rouge-1']['r'] for s in scores])
        best['rouge-1']['p']=max([s['rouge-1']['p'] for s in scores])
        best['rouge-2']['f']=max([s['rouge-2']['f'] for s in scores])
        best['rouge-2']['r']=max([s['rouge-2']['r'] for s in scores])
        best['rouge-2']['p']=max([s['rouge-2']['p'] for s in scores])
        avg=dict()
        avg['rouge-1']=dict()
        avg['rouge-2']=dict()
        avg['rouge-1']['f']=sum([s['rouge-1']['f'] for s in scores])/len(scores)
        avg['rouge-1']['r']=sum([s['rouge-1']['r'] for s in scores])/len(scores)
        avg['rouge-1']['p']=sum([s['rouge-1']['p'] for s in scores])/len(scores)
        avg['rouge-2']['f']=sum([s['rouge-2']['f'] for s in scores])/len(scores)
        avg['rouge-2']['r']=sum([s['rouge-2']['r'] for s in scores])/len(scores)
        avg['rouge-2']['p']=sum([s['rouge-2']['p'] for s in scores])/len(scores)
        
        return best, avg
    
    ###############################################################
    # compute Rouge F1: from this to summary
    ###############################################################
    def computeRougeForText(self, gold, text):
        # self.text to gold.text
        scores = self.Rouge.get_scores(text, gold.text)
        if self.DEBUG>2:
            print("Rouge scores = ",scores)
        return scores
    
    ###############################################################
    # compute Rouge F1: from this to summary
    ###############################################################
    def computeRouge(self, gold):
        # self.text to gold.text
        scores = self.Rouge.get_scores(self.text, gold.text)
        if self.DEBUG>2:
            print("Rouge scores = ",scores)
        return scores
    
    ##############################################################################
    # get R scores for all texts in given list
    ##############################################################################
    @staticmethod
    def getRougeScores(gold, clusters):
        rscores=[]
        rg=Rouge()
        for t in clusters['texts']:
            scores = rg.get_scores(t, gold.text)
            rscores.append(scores)
        return rscores
    
    '''
    ##############################################################################
    # set R scores for all texts in given list
    ##############################################################################
    @staticmethod
    def setRougeScores(gold, clusters):
        rscores=[]
        rg=Rouge()
        if DEBUG>3:
            print("got clusters[texts]: ", clusters['texts'])
        for key in clusters['texts'].keys():
            scores = rg.get_scores(clusters['texts'][key], gold.text)
            clusters['rouge'][key]=scores[0]['rouge-1']['f']
            #clusters['r1-p']=r[0]['rouge-1']['p']
            #clusters['r1-r']=r[0]['rouge-1']['r']
            # later: use all rouge scores
    '''
    
    ##############################################
    # active cluster names
    ##############################################
    @staticmethod
    def get_data_types():
        names = ['sentence-embedding','pos', 'dep', 'tags', 'ner', 'sentiment']
        return names
    
    @staticmethod
    def get_cluster_subtypes():
        names = ['basic', 'ne']
        return names
    
    
    ################################################################################
    # generate node embeddings for
    ################################################################################
    def computeNodeVectors(self):
        self.embedding_graph=dict()
        self.model=dict()
        
        types=Document.get_data_types() 
        for dname in types:
            if self.DEBUG>2:
                print("Generating node2vec for data of type=",dname)
            model, graph = self.generateNodeEmbeddings(dname)
            if self.DEBUG>2:
                print("Done node2vec for data of type=",dname)
          
    ##############################################################################
    # get dependency trees
    ##############################################################################            
    def getDepTreeData(self):
        trees=[]
        for i in range(len(self.sentences)):
            trees.append(self.sentences[i]['nx-tree'])
        
        return trees
    
    
    ##############################################################################
    # get node embedding vector for the dependency tree root
    ##############################################################################            
    def getDepTreeRootEmbeddings(self):
        trees=[]
        for i in range(len(self.sentences)):
            trees.append(self.sentences[i]['dep-tree-root'])
        
        return trees
    
    ##############################################################################
    # get short sentence data: only BERT vector+sentiment data
    ##############################################################################
    def getBertSentenceData(self):
        data=[]
        
        for i in range(len(self.sentences)):
            d1 = self.sentences[i]['sentence-embedding']
            d2 = self.sentences[i]['sentiment']
            if self.DEBUG>2:
                print("Concatenating d1.shape=",d1.shape)
            if d1.shape[0]>1: # take the 1st vector only
                d1=d1[0:1]
                if self.DEBUG>2:
                    print("        fixed d1.shape=",d1.shape)
            d2=np.asarray(d2)
            if self.DEBUG>2:
                print("              d2.shape=",d2.shape)
            d2=np.reshape(d2,(1,len(d2)))
            if self.DEBUG>2:
                print("        fixed d2.shape=",d2.shape)
            d=np.concatenate((d1,d2), axis=1)
            if self.DEBUG>2:
                print("        final  d.shape=",d.shape)
            data.append(d)
        
        return data
    
    ##############################################################################
    # get sentences & doc id
    ##############################################################################  
    def getSentenceListWithDocid(self):
        sdata=[]
        for i in range(len(self.sentences)):
            sd=dict()
            sd['docid']=self.short_name
            sd['sid']=self.sentences[i]['sid']
            sd['text']=self.sentences[i]['sentence']
            sdata.append(sd)
        return sdata
    
    ##############################################################################
    # get node2vec data matrix
    ##############################################################################
    def getSentenceData(self):
        # collect all data for all the sentences
        types=Document.get_data_types()
        self.node2vec_matrix=[]
        for i in range(len(self.sentences)):
            smatrix=[]
            # bert vector
            blen=np.asarray(self.sentences[i]['sentence-embedding']).shape[1]
            bdim=math.ceil(float(blen)/float(NODE2VEC_DIMS))
            bvec=np.zeros(bdim*NODE2VEC_DIMS)
            
            if self.DEBUG>2:
                print('se shape=',np.asarray(self.sentences[i]['sentence-embedding']).shape)
                print("blen=",blen,"bdim=",bdim," bvec length divisible by ",NODE2VEC_DIMS,"=",len(bvec))
            if self.DEBUG>2:
                print('se      =',self.sentences[i]['sentence-embedding'])
            for j1 in range(blen):
                bvec[j1]=self.sentences[i]['sentence-embedding'][0][j1]
            bvec=bvec.reshape(bdim,NODE2VEC_DIMS)
            if self.DEBUG>2:
                print("bvec shape=",bvec.shape)
            if self.DEBUG>2:
                print("bvec =",bvec)
            for j1 in range(len(bvec)):
                smatrix.append(list(bvec[j1])) 
            
            # sentiment vector
            svec=[0]*NODE2VEC_DIMS
            for j in range(len(self.sentences[i]['sentiment'])):
                svec[j]=self.sentences[i]['sentiment'][j]
            if self.DEBUG>2:
                print("svec shape=",np.asarray(svec).shape)
            if self.DEBUG>2:
                print("svec =",svec)
            smatrix.append(svec)
            
            # node embeddings
            for dname in types:
                smatrix.append(self.sentences[i]['node-embedding'][dname])
                if self.DEBUG>1:
                    print("node-embedding data =",self.sentences[i]['node-embedding'][dname])
                if self.DEBUG>2:
                    print("node-embedding data for ",dname," shape=",np.asarray(self.sentences[i]['node-embedding'][dname]).shape)
                    
            if self.DEBUG>2:
                print("  smatrix shape=",np.asarray(smatrix).shape)
                
            self.node2vec_matrix.append(smatrix)
        
            if self.DEBUG>2:
                print("---------------> current final matrix shape=",np.asarray(self.node2vec_matrix).shape)
        return self.node2vec_matrix
    
    ##############################################################################
    # Compute sentence labels from gold summaries: find the closest sentence,
    # rougewise
    ##############################################################################
    def computeSentenceLabels(self, golds):
        for i in range(len(self.sentences)):
            binary=[g.getBestSentenceScore(self.sentences[i]['sentence']) for g in golds]
            
            self.sentences[i]['binary']=max(binary)
            if self.DEBUG>2:
                print("Best binary score of sentence #",i," is ",max(binary))
                
    ######################################################
    # collect and return all sentence labels,
    # zero-padded to max document length
    ######################################################
    def getBinarySentenceLabels(self):
        all_labels=[sent['binary'] for sent in self.sentences] 
        
        return all_labels
    
    ######################################################
    # set summary
    ######################################################
    def setSummary(self, summary):
        self.summary=summary
    
    ######################################################
    # go over all sentences in this doc, find
    # the best F1 score for the input sentence
    ######################################################
    def getBestSentenceScore(self, target_sentence):
        if target_sentence in self.text:
            bin_score=1
        else:
            bin_score=0
        #return max(r1_all),max(p1_all),max(f1_all),bin_score
        return bin_score
                    
    ##############################################################################
    # Summary: array of sentence indexes
    # Label: Rouge score of a summary
    ##############################################################################
    def computeLabel(self, summary_indexes, gold):
        summ_text=""
        for index in summary_indexes:
            if index>=0 and index<len(self.sentences):
                summ_text=summ_text+self.sentences[index]['sentence']+" "
        scores = self.Rouge.get_scores(summ_text, gold.text)
        r1f=scores[0]['rouge-1']['f']
        if self.DEBUG>2:
            print("Created label=", r1f)
        return r1f
    
    ##############################################################################
    # Summary: text
    # Label: Rouge score of a summary
    ##############################################################################
    def computeLabelForText(self, summary, gold):
        rscores=[]
        scores = self.Rouge.get_scores(summary, gold.text)
        r1f=scores[0]['rouge-1']['f']
        if self.DEBUG>2:
            print("Created label=", r1f)
        return r1f
    
    
    ##################################################################################
    # constructing random summaries
    ##################################################################################
    ###########################################################            
    # produce variations
    # mask sentences: 0 or existing ones
    ###########################################################
    def generateSummaryVariations(self, repeatEachSelection=1, maxSentences=-1):
        variations, variationIndexes=[], []
        sentencesToUse=len(self.sentences)
        if maxSentences>=0:
            sentencesToUse=min(maxSentences,len(self.sentences))
        for k in range(repeatEachSelection):
            vcount=0
            for i in range(1, sentencesToUse+1):
                sent_indexes=[]
                summ_text=""
                # get i random sentences from the source document
                for j in range (0,i):
                    sid = random.randint(0, len(self.sentences)-1)
                    sent_indexes.append(sid)
                    summ_text=summ_text+self.sentences[sid]['sentence']+" "
                # create variation
                variations.append(summ_text)
                variationIndexes.append(sent_indexes)
                vcount=vcount+1
            if self.DEBUG>2:
                print("generateSummaryVariations repetition=",k," generated ",vcount," variations")
        return variations, variationIndexes
    
    #############################################################################
    # get summaries created by random token elimination from given
    # indexed summaries
    #############################################################################
    def getTokenLevelVariations(self, summaries, maskTokens=1):
        variations=[]
        for summary in summaries:
            # create variation
            for mask_token_count in range(1,maskTokens+1):
                varsumm=""
                for index in summary:
                    # get sentence tokens
                    tokens=self.sentences[index]['tokens']
                    # get random masking
                    mask=set()
                    for i in range(maskTokens):
                        mask.add(random.randint(0, len(tokens)-1))
                    # go over tokens and build a new sentence
                    newsent=""
                    for i in range(len(tokens)):
                        if i not in mask:
                            newsent=newsent+tokens[i].text+SPACE
                        else: # skip this token
                            if self.DEBUG>1:
                                print("masked token=", tokens[i].text)
                    newsent=newsent.strip()
                    newsent=newsent+EOL
                    if self.DEBUG>2:
                        print("masked sentence=", newsent)
                varsumm=varsumm+newsent
                # add summary to the list
                variations.append(varsumm)
        if self.DEBUG>2:
            print("Created ", len(variations)," token-level variations")
        return variations
        
    ###########################################################################
    # Get summaries on sentence- and token-level
    ###########################################################################
    def getRandomSummaries(self, summaryNum, sentenceNum, maskTokens, sentenceLevel=True, tokenLevel=True):
        summaries=[]
        sentenceSummaries, sentenceSummaryIndexes = self.generateSummaryVariations(summaryNum, sentenceNum)
        if sentenceLevel:
            summaries.extend(sentenceSummaries)
        tokenSummaries=self.getTokenLevelVariations(sentenceSummaryIndexes, maskTokens)
        if tokenLevel:
            summaries.extend(tokenSummaries)
        if self.DEBUG>0:
            print("Generated summaries: s-level ",len(sentenceSummaries)," t-level ",len(tokenSummaries))
        return summaries
        