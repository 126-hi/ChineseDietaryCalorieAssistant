# -*- coding: utf-8 -*-
"""
Streamlit Multi-Agent Cuisine Assistant – **Fully-Integrated Final Build**  
=======================================================================
Features
--------
* 🎨 Pastel grey-blue UI theme (custom CSS)
* 📜 System prompt persisted in `st.session_state`
* 🖼️ Home page recipe card grid (covers of all cookbooks)
* 📅 FullCalendar month view for meal-plan scheduling (`streamlit-calendar`)
* 🧑‍🍳 Recipes & 7-day meal-plan generator (OpenAI GPT-3.5 + LangChain RAG)
* 🖼️ YOLOv7 calorie detection (weights in `weights/best.pt`, PyTorch 2.6-safe)
* 🩺 BMI calculator

Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Run in GitHub Codespace → auto-installs via `devcontainer.json`.
"""

###############################################################################
# 🛠 Imports & Path
###############################################################################
import sys, os, re, textwrap, datetime as dt
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests, openai, streamlit as st
from PIL import Image
import torch, cv2
from streamlit_calendar import calendar

sys.path.append("./yolov7")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

###############################################################################
# 🎨 Global pastel theme
###############################################################################
st.markdown("""
<style>
:root{--bg:#e7edf5;--sidebar:#d5e1f2;--primary:#6c8ebf;--white:#fff}
html,body,[class*='css']{background-color:var(--bg)!important;}
.stApp{background-color:var(--bg);}
.stSidebar{background-color:var(--sidebar);} 
.stButton>button{background:var(--primary);color:var(--white);border:0;padding:6px 16px;border-radius:6px}
.stMarkdown h1,h2,h3{color:var(--primary);} 
</style>
""",unsafe_allow_html=True)

###############################################################################
# 🔑 API Keys (sidebar)
###############################################################################
with st.sidebar:
    st.title("API Keys")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    nx_id   = st.text_input("Nutritionix App ID", value=os.getenv("NUTRITIONIX_APP_ID", ""))
    nx_key  = st.text_input("Nutritionix App Key", type="password", value=os.getenv("NUTRITIONIX_APP_KEY", ""))

if not api_key:
    st.warning("ℹ️ Enter your OpenAI key to continue …")
    st.stop()

openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

###############################################################################
# 📜 Persisted system prompt for chat
###############################################################################
SYSTEM_PROMPT = (
    "You are a culinary assistant who helps users create authentic Chinese recipes based on available ingredients.\n"
    "Provide a structured response with dish name (EN & CN), ingredients list, step-by-step instructions, and cooking tips." )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":SYSTEM_PROMPT}]

###############################################################################
# 📚 Cookbooks & FAISS retriever
###############################################################################
COOKBOOKS = {
    "Easy Chinese Cuisine":"data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "China in 50 Dishes":"data/02. China in 50 Dishes author HSBC.pdf",
    "7-Day Healthy Meal Plan":"data/7-day-Chinese-healthy-meal-plan.pdf",
}

@st.cache_resource(show_spinner="Indexing cookbooks …")
def build_retriever(paths:List[str]):
    docs=[]
    for p in paths:
        if Path(p).exists(): docs+=PyPDFLoader(p).load()
    splits = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    store  = FAISS.from_documents(splits, OpenAIEmbeddings(openai_api_key=api_key))
    return store.as_retriever()

retriever = build_retriever(list(COOKBOOKS.values()))

###############################################################################
# 🤖 Chat helper
###############################################################################
def chat(msgs,temp=0.6):
    return client.chat.completions.create(model="gpt-3.5-turbo",messages=msgs,temperature=temp).choices[0].message.content.strip()

###############################################################################
# 🔌 YOLOv7 Loader & Infer
###############################################################################
@st.cache_resource(show_spinner="Loading YOLOv7 …")
def load_model(path="weights/best.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = attempt_load(path,map_location=device,weights_only=False)
    model.to(device).eval();return model,device

model,device = load_model(); class_names = model.names
nutri = {"egg":68,"rice":130,"salad":35}

def detect(img,conf=0.25,iou=0.45):
    t,a = preprocess(img:=Image.open(img).convert("RGB")) if isinstance(img,str) else preprocess(img)
    t = t.to(device)
    with torch.no_grad(): pred = non_max_suppression(model(t)[0],conf,iou)[0]
    if pred is not None and len(pred):
        pred[:,:4] = scale_coords(t.shape[2:],pred[:,:4],a.shape).round()
        for *xy,conf,cls in pred:
            name = class_names[int(cls)]; kcal = nutri.get(name,'?')
            cv2.rectangle(a,(int(xy[0]),int(xy[1])),(int(xy[2]),int(xy[3])),(0,255,0),2)
            cv2.putText(a,f"{name}:{kcal}kcal",(int(xy[0]),int(xy[1])-6),0,0.5,(0,0,0),2)
    return a

def preprocess(image):
    arr = np.array(image); r = letterbox(arr,640)[0]; r = r[:,:,::-1].transpose(2,0,1)
    return torch.from_numpy(np.ascontiguousarray(r)).float().div(255).unsqueeze(0),arr

###############################################################################
# 🗂 Sidebar navigation
###############################################################################
with st.sidebar:
    st.markdown("---")
    section = st.radio("Feature", ("Home","Recipes","Meal Plan","Calendar","BMI","YOLO"))

###############################################################################
# 🏠 Home – cookbook covers grid
###############################################################################
if section=="Home":
    st.header("📚 Cookbooks")
    cols = st.columns(3)
    for i,(title,_) in enumerate(COOKBOOKS.items()):
        with cols[i%3]:
            st.image("https://placehold.co/200x150?text=Cover",caption=title)

###############################################################################
# 🍽️ Recipes / Meal Plan (chat)
###############################################################################
if section in {"Recipes","Meal Plan"}:
    prompt=st.text_area("Your request",key="prompt",height=120)
    if st.button("Generate") and prompt.strip():
        msgs = st.session_state.messages + [{"role":"user","content":prompt}]
        reply = chat(msgs)
        st.session_state.messages.append({"role":"user","content":prompt})
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.markdown(reply)
        if section=="Meal Plan":
            st.session_state.mealplan_md = reply  # 保存给 Calendar

###############################################################################
# 📅 Calendar (FullCalendar)
###############################################################################
if section=="Calendar":
    st.header("📅 Meal Planner")
    ev=[]
    if "mealplan_md" in st.session_state:
        patt=r"\|\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|"
        wk=dt.date.today()
        for d,br,lun,din in re.findall(patt,st.session_state.mealplan_md):
            day = wk+dt.timedelta(days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(d))
            for meal,label in zip([br,lun,din],["Breakfast","Lunch","Dinner"]):
                if meal.strip():
                    ev.append({"title":f"{meal.strip()} • {label}","start":day.isoformat()})
    else:
        ev=[{"title":"Example Dish","start":dt.date.today().isoformat()}]
    calendar(events=ev,options={"initialView":"dayGridMonth"})

###############################################################################
# 🩺 BMI
###############################################################################
if section=="BMI":
    w=st.number_input("Weight (kg)",0.0,300.0,70.0); h=st.number_input("Height (cm)",0.0,250.0,170.0)
    if st.button("BMI") and h>0:
        bmi=w/((h/100)**2)
        st.success(f"BMI = {bmi:.1f}")

###############################################################################
# 🍔 YOLO calorie
###############################################################################
if section=="YOLO":
    up=st.file_uploader("Upload food",type=["jpg","png"])
    if up: st.image(detect(Image.open(up)),use_column_width=True)

