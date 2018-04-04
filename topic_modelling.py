
# coding: utf-8

# In[1]:

#loading libraries
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger
from nltk.tokenize import word_tokenize


# In[2]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# In[3]:

#loading data
docs = []
indir = "path to folder containing files"
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        log = open(os.path.join(root, f), 'r', encoding='latin1')
        docs.append(log.read())
time_slice = [100, 150, 250]


# In[4]:


 #cleaning the corpus
def clean(d):
    stop_free = " ".join([i for i in d.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(word for word in punc_free.split() if len(word)>4)
    return normalized


# In[5]:


doc_clean = [clean(d).split() for d in docs]
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean] #generating a document-term matrix


# In[6]:

#running the model on the data
ldaseq = ldaseqmodel.LdaSeqModel(doc_term_matrix, id2word=dictionary, time_slice=time_slice, num_topics=5) 


# In[7]:

#printing the topics generated
ldaseq.print_topics(time=0)


# In[8]:

#printing evolution of topics in time slices
ldaseq.print_topic_times(topic=0) 


# In[27]:

#testing the trained model on a document in the corpus
doc = ldaseq.doc_topics(2)
print(doc)


# In[36]:

#testing the model on this research paper
test_doc = []
f = open('/Users/Moukthika/Desktop/ultimate_test.txt','r',encoding='utf8')
test_doc.append(f.read())
#print(test_doc)
for d in test_doc:
    doc_words = d.split(" ")
    #print(doc_words)
    
doc_words = dictionary.doc2bow(doc_words)
doc_words = ldaseq[doc_words]
print (doc_words)

#testing the model on another document not in the corpus
test_doc2=[]
p = open('/Users/Moukthika/Desktop/pdf_extract/99.txt','r',encoding='utf8')
test_doc2.append(p.read())
for d1 in test_doc2:
    doc2_words = d1.split(" ")
doc2_words = dictionary.doc2bow(doc2_words)
doc2_words = ldaseq[doc2_words]
print (doc2_words)


# In[37]:

#comparing the above two documents
hellinger(doc_words, doc2_words)



