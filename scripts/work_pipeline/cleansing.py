import sys
import os
#!pip uninstall -y websocket-client    

import json
import time
from pprint import pprint as pp
import copy
import http.client
import io
from wikidata.client import Client
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.request
import pandas as pd
import pyowm
from datetime import datetime as dt
from operator import itemgetter
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from googletrans import Translator
import re
import emoji
from arango import ArangoClient
from rdflib import Graph, plugin, URIRef, Literal, ConjunctiveGraph, store, Namespace
from rdflib.plugin import register, Parser
#from pyld import jsonld
import joblib as jl
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import numpy as np


try:
    import nltk
except ImportError:
    #!pip install nltk
    import nltk
    nltk.download('stopwords')

from nltk.stem.snowball import SnowballStemmer

try:
    from bs4 import BeautifulSoup 
except ImportError:
    #!pip install bs4 --user
    from bs4 import BeautifulSoup 
    
try:
    from langdetect import detect
except ImportError:
    #!pip install langdetect --user
    from langdetect import detect

# try:
#     from polyglot.detect import Detector
# except ImportError:
#     #!pip install --user pyicu pycld2
#     !pip install polyglot --user

from polyglot.detect import Detector
from translate import Translator
import emoji    
def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

#translator = Translator()
#give_emoji_free_text(<your message>)
#translator.translate(give_emoji_free_text(<your message>), dest = 'en').text

LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
    'fil': 'Filipino',
    'he': 'Hebrew'
}
    
Snowball_languages = (
       "arabic",
       "danish",
       "dutch",
       "english",
       "finnish",
       "french",
       "german",
       "hungarian",
       "italian",
       "norwegian",
       "porter",
       "portuguese",
       "romanian",
       "russian",
       "spanish",
       "swedish",
)

bad_words=["www", "http", "https", "th", "pm", "ticket", "org", "event", "link", "registr",
               "meetup", "event", "group", "regist", "pleas", "please", "join", "rsvp", "member", 
               "venu", "free", "comment", "thank", "attend", "eventbrit", "mr", "st", "rd", "hour", "new", "time", 
               "boston", "like"   ]
#     def clean_description(desc):

def clean_description(desc):
    #print("yolo")
    stemm=False
    try:
        #html_less = BeautifulSoup(desc).get_text() 
        html_less=BeautifulSoup(desc,"lxml").text
    except Exception as e:
        #print(e)
        html_less=str(desc)
    #rm punctuation
    #for lang in Detector(html_less).languages:
    #    print(lang)
    html_less= re.sub(r'^https?:\/\/.*[\r\n]*', '', html_less, flags=re.MULTILINE)
    
    no_emoj=give_emoji_free_text(html_less)
    
    try:
        langs=Detector(no_emoj).languages
        lang=langs[0].name.lower()        
    except:
        print(no_emoj,"--",desc)
        stemm=False
        return "Error"
        
    try:     
        #translator = Translator()
        #trans = translator.translate(no_emoj, dest = 'en')
        #no_emoj=trans.text
        #lang = LANGUAGES[trans.src]
        translator= Translator(from_lang=langs[0].code, to_lang="en")
        no_emoj = translator.translate(no_emoj)
        if no_emoj== 'MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY. NEXT AVAILABLE IN  13 HOURS 32 MINUTES 18 SECONDSVISIT HTTPS://MYMEMORY.TRANSLATED.NET/DOC/USAGELIMITS.PHP TO TRANSLATE MORE':
            print("ERROR")
    except Exception as e:
        print(e)
        return "Error"
    
    no_punct = re.sub("[^a-zA-Z]", " ", no_emoj)
    #no_punct=no_punct.replace("|","").replace("!","").replace("?","")
    

    tokens = no_punct.lower().split()
    
    #nltk supported lang
    europ_languages = ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish']
    #europ_languages=nltk.corpus.stopwords.fileids()
    
    no_tokens = set(nltk.corpus.stopwords.words(europ_languages))                  
    #print(my_stop_words)
    no_stop_tokens = [token for token in tokens if  token not in no_tokens]   
    #print(no_stop_tokens)
    if stemm:
        if lang in Snowball_languages:
            Stemmer=SnowballStemmer(lang.lower())
            stemmed=[Stemmer.stem(elem) for elem in no_stop_tokens]
            good_words=[elem for elem in stemmed if elem not in bad_words]
            return ' '.join(good_words)
        #return( " ".join(stemmed))
            
    good_words=[elem for elem in no_stop_tokens if elem not in bad_words]
    return ' '.join(good_words)    
    #return( " ".join(no_stop_tokens))
    
def vec_for_learning(model, tagged_docs):
    #sents = tagged_docs.values
    regressors = [model.infer_vector(doc.words, steps=20) for doc in tqdm(tagged_docs.values)]
    return  regressors

def fake_tagged_doc(desc):
    arr=np.asarray(desc)
    arr=pd.Series(arr)

    test_tagged = arr.apply( lambda r: TaggedDocument(words=str(r).split(" "), tags=["NaN"]))#, axis=1)
    return test_tagged
    
