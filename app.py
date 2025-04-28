# -*- coding: utf-8 -*-
"""
Streamlit Multi‑Agent Cuisine Assistant — Minimal Meal‑Plan Edition
==================================================================
This version drops the standalone Calendar and focuses on a clean
**Meal‑Plan‑only** workflow that returns a 7‑day plan as a table.
"""

###############################################################################
# 🔧 Imports & Path
###############################################################################
import sys, os, re, datetime as dt, io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import openai, torch, cv2
from PIL import Image
from streamlit_calendar import calendar   # still required by YOLO widget icons

sys.path.append("./yolov7")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

###############################################################################
# 🌈 UI Theme
###############################################################################
st.markdown("""
<style>
:root{--bg:#e7edf5;--sidebar:#d5e1f2;--primary:#6c8ebf;--white:#fff}
html,body,[class*='css']{background-color:var(--bg)!important;}
.stApp{background-color:var(--bg);}
.stSidebar{background-color:var(--sidebar);} 
.stButton>button{background:var(--primary);color:var(--white);border:0;padding:6px 16px;border-radius:12px;font-weight:bold}
.stButton>button:hover{background:#5a7bad;color:white}
.stMarkdown h1,h2,h3{color:var(--primary);} 
</style>
""", unsafe_allow_html=True)

###############################################################################
# 🏷️  Title
###############################################################################
st.markdown("""
<h1 style='text-align: center'>🥢 Chinese Cuisine & Nutrition Assistant</h1>
<p style='text-align: center;color:grey;'>Your AI‑powered Chinese Meal‑Planner & Food Tracker</p>
""", unsafe_allow_html=True)

###############################################################################
# 🔑  API Key
###############################################################################
with st.sidebar:
    st.title("API Key")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

if not api_key:
    st.warning("🔑 Please enter your OpenAI key to continue …")
    st.stop()

openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

###############################################################################
# 📚  Cookbook Retrieval (RAG)
###############################################################################
COOKBOOKS = {
    "Easy Chinese Cuisine":"data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "China in 50 Dishes":"data/02. China in 50 Dishes author HSBC.pdf",
    "7‑Day Healthy Meal Plan":"data/7-day-Chinese-healthy-meal-plan.pdf",
}

@st.cache_resource(show_spinner="Indexing cookbooks …")
def build_retriever(paths:List[str]):
    docs=[]
    for p in paths:
        if Path(p).exists():
            docs+=PyPDFLoader(p).load()
    chunks = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    store  = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=api_key))
    return store.as_retriever()

retriever = build_retriever(list(COOKBOOKS.values()))

def chat_with_cookbook(query:str, k:int=3, temp:float=0.6)->str:
    ctx = "\n\n".join(d.page_content for d in retriever.get_relevant_documents(query,k=k))
    prompt = (
        "You are a Chinese‑cuisine assistant. Using only the context below, "
        "answer the user.\n\nContext:\n"+ctx+"\n\nUser request:\n"+query)
    return client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],temperature=temp).choices[0].message.content.strip()

###############################################################################
# 🗂️  Sidebar Navigation (Calendar removed)
###############################################################################
with st.sidebar:
    section = st.radio("Feature", (
        "🏠 Home", "🍽️ Recipes", "📅 Meal Plan", "⚖️ BMI", "📷 YOLO", "🥗 Nutrition"))

###############################################################################
# 🏠 Home
###############################################################################
if section=="🏠 Home":
    st.header("🏠 Welcome")
    st.write("Use the sidebar to explore Recipes, Meal Plans and Nutrition tracking.")

###############################################################################
# 🍽️ Recipe Generator (unchanged, RAG‑enhanced)
###############################################################################
if section=="🍽️ Recipes":
    st.header("🍽️ Recipe Generator")
    desc = st.text_area("Describe the dish you want:")
    if st.button("Generate Recipe") and desc:
        st.markdown(chat_with_cookbook(desc))

###############################################################################
# 📅 Meal‑Plan  (table only, no calendar)
###############################################################################
if section=="📅 Meal Plan":
    st.header("📅 7‑Day Meal Plan")

    ing = st.text_input("Ingredients available:", "tofu, beef, broccoli, garlic, egg")
    kcal = st.number_input("Target calories / day",500,4000,1500)

    if st.button("Generate Meal Plan"):
        q = (f"Create a 7‑day Chinese meal plan using: {ing}. "
             f"Each day ≈ {kcal} kcal. "
             "Respond **only** as a markdown table in the format:\n"
             "Day | Dish | Ingredients | Calories")
        md = chat_with_cookbook(q)
        st.session_state.mealplan_md = md

    if "mealplan_md" in st.session_state:
        st.markdown(st.session_state.mealplan_md)
        # also render as dataframe for filtering / csv
        table_lines=[l for l in st.session_state.mealplan_md.splitlines() if "|" in l]
        if len(table_lines)>=2:
            csv_text="\n".join([re.sub(r"^\||\|$","",l) for l in table_lines])
            df=pd.read_csv(io.StringIO(csv_text),sep="|",engine="python").rename(columns=str.strip)
            st.dataframe(df,use_container_width=True)

###############################################################################
# ⚖️ BMI
###############################################################################
if section=="⚖️ BMI":
    st.header("⚖️ BMI Calculator")
    w=st.number_input("Weight (kg)",0.0,250.0,70.0)
    h=st.number_input("Height (cm)",0.0,230.0,170.0)
    if st.button("Calculate") and h>0:
        st.success(f"BMI = {w/((h/100)**2):.1f}")

###############################################################################
# 📷 YOLO Calorie (kept minimal, unchanged)
###############################################################################
if section=="📷 YOLO":
    st.header("📷 Estimate Calories from Image")
    up=st.file_uploader("Upload food image",["jpg","png"])
    if up:
        img_arr=detect(Image.open(up))
        st.image(Image.fromarray(img_arr),use_column_width=True)

###############################################################################
# 🥗 Nutrition Tracker (same as previous, emoji table)
###############################################################################
if section=="🥗 Nutrition":
    st.header("🥗 Food Nutrition Tracker")
    today=dt.date.today().isoformat()
    if "food_log" not in st.session_state:
        st.session_state.food_log={}
    st.session_state.food_log.setdefault(today,[])

    food_db={"egg":{"c":68,"p":6,"carb":1,"fat":5},"rice":{"c":130,"p":2.7,"carb":28,"fat":0.3},
             "salad":{"c":35,"p":2,"carb":7,"fat":0.2},"chicken breast":{"c":165,"p":31,"carb":0,"fat":3.6},
             "apple":{"c":95,"p":0.5,"carb":25,"fat":0.3}}
    emoji={"egg":"🍳","rice":"🍚","salad":"🥗","chicken breast":"🍗","apple":"🍎"}

    f=st.selectbox("Food",list(food_db.keys()))
    q=st.number_input("Servings",1,10,1)
    if st.button("Add"):
        item=food_db[f].copy(); item.update(name=f"{emoji.get(f,'')} {f}",qty=q)
        st.session_state.food_log[today].append(item)

    log=st.session_state.food_log[today]
    if log:
        df=pd.DataFrame(log)
        df["cal"]=df["c"]*df["qty"]
        df["prot"]=df["p"]*df["qty"]
        st.dataframe(df[["name","qty","cal","prot"]])
        st.success(f"Total {df['cal'].sum():.0f} kcal, Protein {df['prot'].sum():.1f} g")
      





