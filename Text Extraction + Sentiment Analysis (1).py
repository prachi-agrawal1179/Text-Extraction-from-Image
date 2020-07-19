#!/usr/bin/env python
# coding: utf-8

# In[49]:


from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import argparse
import cv2
import os
img = cv2.imread(r'C:\Users\mypc\Downloads\755275_1304907_bundle_archive\Love is Love\Dataset\Test119.jpg',0)


# In[50]:


filename = "{}.png".format(os.getpid())
print(filename)

cv2.imwrite(filename, img)


# In[51]:


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)


# In[35]:


import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
tokenized_word=sent_tokenize(text)
tokenized_text=word_tokenize(text)
print(tokenized_text)


# In[36]:


from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)


# In[9]:


fdist.most_common(2)


# In[10]:


import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()


# In[11]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[12]:


filtered_sent=[]
for w in tokenized_text:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_text)
print("Filterd Sentence:",filtered_sent)


# In[13]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[15]:


tokens=nltk.word_tokenize(text)
print(tokens)


# In[83]:


nltk.pos_tag(tokens)


# In[ ]:




