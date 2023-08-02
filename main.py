from typing import Union
import pickle
import numpy as np
import pandas as pd
import ast
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fastapi import FastAPI

app = FastAPI()


lm=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

#loading the ML model
model_pickle = open("./rc_model.pkl", "rb")
clf = pickle.load(model_pickle)

tfidf = open("./tfidf.pkl", "rb")
vect = pickle.load(tfidf)

def preprocess_inf(text):
  def extract(x):
    if x['employment_history'][1:-1] == '':
      return ''
    s=ast.literal_eval(x['employment_history'][1:-1])
    try:
      op=str(s['designation'])+" "+str(s['role'])+' '+str(s['keyskills'])
      return op
    except TypeError:
      s=x['employment_history'][1:-1]
      string = s.replace('"', "'")
      string = "[" + string + "]"
      try:
        list_of_dicts = ast.literal_eval(string)
      except SyntaxError:
        string = "[" + s + "]"
        list_of_dicts = ast.literal_eval(string)
        l=[]
        for i in list_of_dicts:
          l.append(str(i['designation'])+" "+str(i['role'])+' '+str(i['keyskills']))
        return l
      ls=[]
      for i in list_of_dicts:
        ls.append(str(i['designation'])+" "+str(i['role'])+' '+str(i['keyskills']))
      return ls
  def spl(x):
    ls=[]
    string=x
    # Replace double quotes with single quotes
    string = string.replace('"', "'")

    # Add square brackets to make it a list
    string = "[" + string + "]"

    # Convert the string into a list of dictionaries
    list_of_dicts = ast.literal_eval(string)

    for i in list_of_dicts:
        if i['experience']=='':
            continue
        if float(i["experience"])>0.0:
            ls.append(i)
    return ls
  eh= extract(text)
  exp=str(text['overall_experience'])
  skill=text['skillset'][2:-3]
  a=text['domain_with_experience'][1:-1]
  b=spl(a)
  c=[str(x) for x in b]
  d=[x[3:-2] for x in c]
  dwe=str(d)
  ach=text['achievements'][2:-2]
  cert=text['certifications'][2:-2]
  hb=text['hobby'][1:-1]
  extracted=str(eh)+" "+str(exp)+" "+str(skill)+' '+str(dwe)+" "+str(ach)+" "+str(cert)+' '+str(hb)
  extracted=re.sub('[%s]' % re.escape(string.punctuation), '' , extracted)
  extracted=re.sub(r"\d+", "", extracted)
  def clean_text(text):
    text=text.lower()
    words=nltk.word_tokenize(text)
    words=[lm.lemmatize(x) for x in words]
    words=" ".join([x if x not in stop_words else'' for x in words])
    return words
  final_text=clean_text(extracted)
  return final_text

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def prediction(res: dict):
  txt=preprocess_inf(res)
  txt_v=vect.transform([txt])
  ans=clf.predict(txt_v)
  ans=ans.tolist()
  return ans
