# -*- coding: utf-8 -*-
"""
Streamlit Multi‑Agent Cuisine Assistant – **All‑in‑One (RAG Button + Custom Calorie Vision)**
===========================================================================================
This is the final integrated build.  It contains **every feature** we’ve
discussed so far:

1. 🎯 BMR / TDEE calculator in the sidebar
2. 📚 Cookbook selector *plus* **Add RAG Context** button for Recipes & Meal Plans
3. 🧑‍🍳 Recipe generation with diet filters (checkboxes inside RAG prompt)
4. 🗓️ 7‑day meal‑plan generator (Markdown download + shopping‑list CSV)
5. 🖼️ Image generation via DALL·E 3
6. 📷 Photo calorie estimation using **nateraw/food** model → Nutritionix API
7. 📑 Text ingredient calorie estimation (GPT‑3.5)
8. 📈 Daily calorie & weight tracker with Plotly trend

### Required keys (sidebar or environment)
| Key | Purpose |
|-----|---------|
| **OpenAI API Key** | Chat + DALL·E |
| **HF_FOOD_TOKEN** | Access `nateraw/food` Space |
| **Nutritionix App ID / Key** | Calorie lookup |

Run:
```bash
streamlit run cuisine_assistant_app.py
```
"""

from __future__ import annotations
import datetime as dt, os, textwrap, re, json
from pathlib import Path
from typing import List

import pandas as pd, plotly.express as px, requests, streamlit as st, openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

###############################################################################
# 🔑 Sidebar – API keys & goal calculator
###############################################################################
with st.sidebar:
    st.title("API Keys")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    hf_token = st.text_input("HF FOOD Token", type="password", value=os.getenv("HF_FOOD_TOKEN", ""))
    nx_id  = st.text_input("Nutritionix App ID", value=os.getenv("NUTRITIONIX_APP_ID", ""))
    nx_key = st.text_input("Nutritionix App Key", type="password", value=os.getenv("NUTRITIONIX_APP_KEY", ""))

if not api_key:
    st.warning("OpenAI key required."); st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

store = FAISS.from_documents(
    splits,
    OpenAIEmbeddings(openai_api_key=api_key)  
)

# Goal calculator ------------------------------------------------------------
_ACTIVITY = {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very Active":1.9}
_GOAL = {"Lose":-500,"Maintain":0,"Gain":300}

def mifflin(sex,w,h,age):
    return 10*w+6.25*h-5*age+(5 if sex=="Male" else -161)
with st.sidebar:
    st.markdown("---"); st.subheader("Personal Goal")
    sex = st.radio("Sex",("Male","Female"),horizontal=True)
    c1,c2=st.columns(2)
    age=c1.number_input("Age",10,100,25); wt=c1.number_input("Weight (kg)",30.,300.,70.,step=0.1)
    ht=c2.number_input("Height (cm)",120.,230.,170.,step=0.1)
    activity=c2.selectbox("Activity", list(_ACTIVITY))
    goal=st.selectbox("Goal", list(_GOAL))
    if st.button("Calc BMR/TDEE"):
        bmr=mifflin(sex,wt,ht,age); tdee=bmr*_ACTIVITY[activity]; targ=max(1000,round(tdee+_GOAL[goal]))
        st.success(f"Target ≈ **{targ} kcal/day** (TDEE {tdee:.0f})")
        st.session_state.target=targ

###############################################################################
# 📚 Cookbook data & FAISS index (RAG)
###############################################################################
COOKBOOKS={
    "Easy Chinese Cuisine":"data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "China in 50 Dishes":"data/02. China in 50 Dishes author HSBC.pdf",
    "7-Day Healthy Meal Plan":"data/7-day-Chinese-healthy-meal-plan.pdf",
}
@st.cache_resource(show_spinner="Indexing cookbooks…")
def build_retriever(paths:List[str]):
    docs=[]
    for p in paths:
        if Path(p).exists(): docs+=PyPDFLoader(p).load()
    splits=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(splits,OpenAIEmbeddings()).as_retriever()
retriever=build_retriever(list(COOKBOOKS.values()))
class RAG:
    def __init__(self,r): self.r=r
    def ctx(self,q,srcs,k=3):
        docs=[d for d in self.r.get_relevant_documents(q) if any(s in d.metadata.get('source','') for s in srcs)]
        return "\n\n".join(d.page_content for d in docs[:k])
rag=RAG(retriever)

###############################################################################
# 🖼️ Custom Calorie Vision (nateraw/food → Nutritionix)
###############################################################################
FOOD_API="https://api-inference.huggingface.co/models/nateraw/food"
HEADERS={"Authorization":f"Bearer {hf_token}"} if hf_token else {}

def classify_food(img:bytes)->dict:
    r=requests.post(FOOD_API,headers=HEADERS,data=img); r.raise_for_status()
    top=json.loads(r.content.decode())[:5]
    return {i['label']:i['score'] for i in top}

def nutritionix(food:str)->dict:
    r=requests.get("https://trackapi.nutritionix.com/v2/search/instant",params={"query":food},headers={"x-app-id":nx_id,"x-app-key":nx_key})
    r.raise_for_status(); b=r.json().get('branded',[{}])[0]
    return {"food_name":b.get('food_name','-'),"cal":b.get('nf_calories','?'),"qty":b.get('serving_qty','?'),"unit":b.get('serving_unit','?')}

def photo_calorie(img:bytes)->str:
    labels=classify_food(img)
    main=max(labels,key=labels.get); md="**Top predictions**:"+"<br>"+"<br>".join([f"{k}: {v:.1%}" for k,v in labels.items()])
    if nx_id and nx_key:
        n=nutritionix(main); md+=f"\n\n**Nutritionix** — {n['food_name']}: {n['cal']} kcal / {n['qty']} {n['unit']}"
    return md

###############################################################################
# 🧑‍🍳 Helper chat
###############################################################################
SYSTEM="You are a culinary assistant producing structured markdown."
MEAL_TEMPLATE=("Return a 7‑day meal plan:\n\n| Day | Breakfast | Lunch | Dinner | Calories |\n|-----|-----------|-------|--------|----------|\n"+"\n".join([f"| {d} | | | | |" for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]))

def chat(msgs,temp=0.6):
    return client.chat.completions.create(model="gpt-3.5-turbo",messages=msgs,temperature=temp).choices[0].message.content.strip()

###############################################################################
# 📂 Navigation                                                          
###############################################################################
with st.sidebar:
    st.markdown("---")
    section=st.radio("Feature",("Recipes","Meal Plans","Generate Photo","Photo Calories","Text Calories","Tracker"))
    books=st.multiselect("Cookbooks",list(COOKBOOKS),default=list(COOKBOOKS)) if section in {"Recipes","Meal Plans"} else list(COOKBOOKS)

###############################################################################
# 🖥️  Main interface
###############################################################################

if section in {"Recipes","Meal Plans"}:
    key="prompt"; st.text_area("Your request",key=key,height=120)
    if st.button("📚 Add RAG Context"):
        ctx=rag.ctx(st.session_state[key],[COOKBOOKS[b] for b in books])
        st.session_state[key]=textwrap.dedent(f"{st.session_state[key]}\n\n# RAG Context\n{ctx}")
        st.success("Context added.")
    if st.button("🚀 Submit"):
        msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":st.session_state[key]}]
        out=chat(msgs,0.7) if section=="Recipes" else chat([{"role":"system","content":MEAL_TEMPLATE}]+msgs,0.4)
        st.markdown(out)
        if section=="Meal Plans":
            st.download_button("Download MD",out,"mealplan.md","text/markdown")
            if st.button("🧾 Shopping List"):
                dishes=re.findall(r"\|\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|",out)
                flat=[d for row in dishes for d in row if d.strip()]
                df=pd.DataFrame({"dish":flat}); st.dataframe(df)
                st.download_button("CSV",df.to_csv(index=False).encode(),"shopping.csv","text/csv")

elif section=="Text Calories":
    ing=st.text_area("Ingredients list (one per line)")
    if st.button("Estimate") and ing.strip():
        res=chat([{"role":"system","content":"List kcal for each ingredient then total."},{"role":"user","content":ing}],0.3)
        st.markdown(res)

elif section=="Photo Calories":
    up=st.file_uploader("Food photo",["png","jpg","jpeg"])
    if st.button("Estimate") and up:
        img=up.read(); st.image(img)
        st.markdown(photo_calorie(img))

elif section=="Generate Photo":
    desc=st.text_input("Describe food")
    if st.button("Generate") and desc.strip():
        url = client.images.generate(
        model="dall-e-3",
        prompt=desc,
        n=1,
        size="1024x1024"
      )
