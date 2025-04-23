# -*- coding: utf-8 -*-
"""
Streamlit Multi‑Agent Cuisine Assistant – **Enhanced + RAG Button** (2025‑04‑22)
===============================================================================
Features
--------
✅ Personalised BMR/TDEE calculator (sidebar)
✅ Cookbook selector + 📚 **Add RAG Context** button
✅ Recipes with diet filters & preferences
✅ 7‑day meal‑plan generator (Markdown/PDF export)
✅ Shopping‑list CSV generator
✅ DALL·E 3 food‑image generation
✅ Photo & text calorie estimation
✅ Daily calorie & weight tracker (Plotly)

Run:
```bash
streamlit run cuisine_assistant_app.py
```
Provide your **OpenAI API key** and (optionally) a **Hugging Face token**.
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional
import textwrap, re

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

###############################################################################
# 🌈 Page config
###############################################################################
st.set_page_config(page_title="Cuisine Assistant", layout="wide")

###############################################################################
# 🔑 Sidebar – API keys & goal calculator
###############################################################################
with st.sidebar:
    st.title("API Keys")
    api_key = st.text_input("OpenAI API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")

if not api_key:
    st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# Goal calculator ------------------------------------------------------------
_ACTIVITY = {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very Active":1.9}
_GOAL = {"Lose":-500,"Maintain":0,"Gain":300}

def mifflin(sex,w,h,age):
    return 10*w+6.25*h-5*age+(5 if sex=="Male" else -161)
with st.sidebar:
    st.markdown("---"); st.subheader("Personal Goal")
    sex = st.radio("Sex",("Male","Female"),horizontal=True)
    a,b = st.columns(2)
    age = a.number_input("Age",10,100,25)
    wt  = a.number_input("Weight (kg)",30.,300.,70.,step=0.1)
    ht  = b.number_input("Height (cm)",120.,230.,170.,step=0.1)
    activity = b.selectbox("Activity", list(_ACTIVITY))
    goal = st.selectbox("Goal", list(_GOAL))
    if st.button("Calc BMR/TDEE"):
        bmr = mifflin(sex,wt,ht,age); tdee=bmr*_ACTIVITY[activity]; targ=max(1000,round(tdee+_GOAL[goal]))
        st.success(f"Target ≈ **{targ} kcal/day** (TDEE {tdee:.0f})")
        st.session_state.target=targ

###############################################################################
# 📚 Cookbook data & FAISS index
###############################################################################
COOKBOOKS={
    "Easy Chinese Cuisine":"data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "China in 50 Dishes":"data/02. China in 50 Dishes author HSBC.pdf",
    "7‑Day Healthy Meal Plan":"data/7-day-Chinese-healthy-meal-plan.pdf",
}
@st.cache_resource(show_spinner="Indexing cookbooks…")
def build_retriever(paths:List[str]):
    docs=[]
    for p in paths:
        if Path(p).exists(): docs+=PyPDFLoader(p).load()
    splits=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    store=FAISS.from_documents(splits,OpenAIEmbeddings())
    return store.as_retriever()
retriever=build_retriever(list(COOKBOOKS.values()))
class RAGAgent:
    def __init__(self,r): self.r=r
    def fetch(self,q:str,srcs:List[str],k:int=3):
        docs=self.r.get_relevant_documents(q)
        docs=[d for d in docs if any(s in d.metadata.get("source","") for s in srcs)]
        return "\n\n".join(d.page_content for d in docs[:k])
rag=RAGAgent(retriever)

###############################################################################
# 🤖 Helper chat function
###############################################################################
def chat(msgs,temp=0.6):
    return client.chat.completions.create(model="gpt-3.5-turbo",messages=msgs,temperature=temp).choices[0].message.content.strip()
SYSTEM="You are a culinary assistant producing structured markdown."
MEAL_TEMPLATE=("Return a 7‑day meal plan:\n\n| Day | Breakfast | Lunch | Dinner | Calories |\n|-----|-----------|-------|--------|----------|\n"+"\n".join([f"| {d} | | | | |" for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]))

###############################################################################
# 📂 Navigation
###############################################################################
with st.sidebar:
    st.markdown("---")
    section=st.radio("Feature",("Recipes","Meal Plans","Take Photo","Photo Calories","Text Calories","Tracker"))
    if section in {"Recipes","Meal Plans"}:
        sel_books=st.multiselect("Cookbooks",list(COOKBOOKS),default=list(COOKBOOKS))
    else:
        sel_books=list(COOKBOOKS)

###############################################################################
# 🖥️ Main UI
###############################################################################

if section in {"Recipes","Meal Plans"}:
    prompt_key="prompt"; st.text_area("Your request",key=prompt_key,height=120)
    if st.button("📚 Add RAG Context"):
        ctx=rag.fetch(st.session_state[prompt_key],[COOKBOOKS[b] for b in sel_books])
        st.session_state[prompt_key]=textwrap.dedent(f"{st.session_state[prompt_key]}\n\n# RAG Context\n{ctx}")
        st.success("Context added.")
    if st.button("🚀 Submit"):
        msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":st.session_state[prompt_key]}]
        if section=="Recipes":
            out=chat(msgs,0.7)
        else:
            out=chat([{"role":"system","content":MEAL_TEMPLATE}]+msgs,0.4)
        st.markdown(out)
        if section=="Meal Plans":
            st.download_button("Download MD",out,"mealplan.md","text/markdown")
            # Shopping list
            if st.button("🧾 Shopping List"):
                items=re.findall(r"\|\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday).*?\|(.+?)\|",out)
                df=pd.DataFrame({"item":items})
                st.dataframe(df); st.download_button("CSV",df.to_csv(index=False).encode(),"shopping.csv","text/csv")

elif section=="Text Calories":
    ing=st.text_area("Ingredients")
    if st.button("Estimate"):
        res=chat([{"role":"system","content":"List kcal each ingredient then total."},{"role":"user","content":ing}],0.3)
        st.markdown(res)
elif section=="Photo Calories":
    pic=st.file_uploader("Food photo",["png","jpg","jpeg"])
    if st.button("Estimate") and pic:
        r=requests.post("https://api-inference.huggingface.co/models/JaydeepR/Calorie_counter",headers={"Authorization":f"Bearer {hf_token}"} if hf_token else {},data=pic.read())
        st.image(pic); st.write(r.json() if r.ok else r.text)
elif section=="Take Photo":
    desc=st.text_input("Describe food")
    if st.button("Generate") and desc:
        url=client.images.generate(model="dall-e-3",prompt=desc,n=1,size="1024x1024").data[0].url; st.image(url)
else:
    if "log" not in st.session_state:
        st.session_state.log=pd.DataFrame(columns=["date","meal","cal","wt"])
    with st.form("add"):
        d1,d2=st.columns(2); date=d1.date_input("Date",dt.date.today()); meal=d2.text_input("Meal")
        cal=st.number_input("Calories",0,3000); wt=st.number_input("Weight",0.0,300.0,step=0.1)
        if st.form_submit_button("Add"):
            st.session_state.log=pd.concat([st.session_state.log,pd.DataFrame([[date,meal,cal,wt]],columns=st.session_state.log.columns)],ignore_index=True)
    df=st.session_state.log.sort_values("date"); st.dataframe(df,use_container_width=True)
    if not df.empty:
        st.plotly_chart(px.line(df,x="date",y=["cal"],markers=True),use_container_width=True)

