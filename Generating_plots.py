
# coding: utf-8

# In[1]:

#importind libraries

import os
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sklearn.feature_extraction.text as text
import numpy as np
from sklearn import decomposition
import gensim
from gensim import corpora


# In[3]:

#loading data
docs = []
indir = "path to folder containing pdf files"
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        log = open(os.path.join(root, f), 'r', encoding='latin1')
        docs.append(log.read())


# In[4]:


vectorizer = text.CountVectorizer(input='docs', stop_words='english', min_df=20)


# In[5]:

#creating a document term matrix
dtm = vectorizer.fit_transform(docs).toarray()


# In[6]:


#computing the vocabulary
vocab = np.array(vectorizer.get_feature_names())


# In[7]:

#printing document and vocab length
dtm.shape


# In[8]:


len(vocab)


# In[10]:



num_topics = 5


# In[11]:


num_top_words = 20



# In[12]:

#creating a model 
clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)


# In[13]:


topic_words = []


# In[14]:

#extracting top words
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])



# In[15]:

#estimating topics in the document
np.seterr(divide='ignore', invalid='ignore')
doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)


# In[16]:


doc_names = []


# In[17]:


for fn in filenames:
    basename = os.path.basename(fn)
    name, ext = os.path.splitext(basename)
    doc_names.append(name)


# In[18]:


doc_names = np.asarray(doc_names)


# In[19]:


doctopic_orig = doctopic.copy()


# In[20]:


num_groups = len(set(doc_names))


# In[21]:



doctopic_grouped = np.zeros((num_groups, num_topics))


# In[22]:


for i, name in enumerate(sorted(set(doc_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[doc_names == name, :], axis=0)


# In[23]:


doctopic = doctopic_grouped


# In[24]:


documents = sorted(set(doc_names))


# In[26]:


for i in range(len(doctopic)):
    top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
    top_topics_str = ' '.join(str(t) for t in top_topics)
    print("{}: {}".format(documents[i], top_topics_str))


# In[27]:


for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))


# In[28]:


filenames


# In[29]:


doctopic.shape


# In[30]:


doctopic


# In[31]:


import matplotlib.pyplot as plt
N, K = doctopic.shape
ind = np.arange(N)
width = 0.5

plt.bar(ind, doctopic[:,0], width=width)
plt.xticks(ind + width/2, filenames)
plt.title('Share of Topic #0')


# In[32]:

#generating the plot
plt.show()


# In[33]:


plots = []
height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, doctopic[:, k], width, color=color)
    else:
        p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
        height_cumulative += doctopic[:, k]
        plots.append(p)


# In[34]:


plt.ylim((0, 1))
plt.ylabel('Topics')
plt.title('Topics in docs')
plt.xticks(ind+width/2, filenames)
plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels)
plt.show()

