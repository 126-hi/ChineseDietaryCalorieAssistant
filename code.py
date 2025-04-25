# -*- coding: utf-8 -*-
"""
Streamlit Multi-Agent Cuisine Assistant – FINAL REBUILD (YOLOv7 + BMI)
=======================================================================
Includes BMI Calculator and integrated YOLOv7 calorie detection (no Gradio)

Run:
```bash
streamlit run cuisine_assistant_app_final.py
```
"""

import os, textwrap, re, json
from pathlib import Path
from typing import List

import pandas as pd
import requests
import streamlit as st
import openai
from PIL import Image
import numpy as np
import torch
import cv2
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

###############################################################################
# 🔑 Sidebar – API keys & goal calculator
###############################################################################
with st.sidebar:
    st.title("API Keys")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    nx_id  = st.text_input("Nutritionix App ID", value=os.getenv("NUTRITIONIX_APP_ID", ""))
    nx_key = st.text_input("Nutritionix App Key", type="password", value=os.getenv("NUTRITIONIX_APP_KEY", ""))

if not api_key:
    st.warning("OpenAI key required."); st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# Goal calculator ------------------------------------------------------------
_ACTIVITY = {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very Active":1.9}
_GOAL = {"Lose":-500,"Maintain":0,"Gain":300}

def mifflin(sex,w,h,age):
    return 10*w + 6.25*h - 5*age + (5 if sex=="Male" else -161)

with st.sidebar:
    st.markdown("---"); st.subheader("Personal Goal")
    sex = st.radio("Sex", ("Male","Female"), horizontal=True)
    c1,c2 = st.columns(2)
    age = c1.number_input("Age", 10, 100, 25)
    wt  = c1.number_input("Weight (kg)", 30., 300., 70., step=0.1)
    ht  = c2.number_input("Height (cm)", 120., 230., 170., step=0.1)
    activity = c2.selectbox("Activity", list(_ACTIVITY))
    goal = st.selectbox("Goal", list(_GOAL))
    if st.button("Calc BMR/TDEE"):
        bmr = mifflin(sex, wt, ht, age)
        tdee = bmr * _ACTIVITY[activity]
        targ = max(1000, round(tdee + _GOAL[goal]))
        st.success(f"Target ≈ **{targ} kcal/day** (TDEE {tdee:.0f})")
        st.session_state.target = targ

###############################################################################
# 📚 Cookbooks & FAISS index (RAG)
###############################################################################
COOKBOOKS = {
    "Easy Chinese Cuisine": "data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "China in 50 Dishes": "data/02. China in 50 Dishes author HSBC.pdf",
    "7-Day Healthy Meal Plan": "data/7-day-Chinese-healthy-meal-plan.pdf",
}
@st.cache_resource(show_spinner="Indexing cookbooks…")
def build_retriever(paths: List[str], api_key: str):
    docs = []
    for p in paths:
        if Path(p).exists():
            docs.extend(PyPDFLoader(p).load())
    splits = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    store = FAISS.from_documents(splits, OpenAIEmbeddings(openai_api_key=api_key))
    return store.as_retriever()

retriever = build_retriever(list(COOKBOOKS.values()), api_key)
class RAG:
    def __init__(self, r): self.r = r
    def ctx(self, q, srcs, k=3):
        docs = [d for d in self.r.get_relevant_documents(q) if any(s in d.metadata.get('source','') for s in srcs)]
        return "\n\n".join(d.page_content for d in docs[:k])
rag = RAG(retriever)

###############################################################################
# 🤖 Chat helper & templates
###############################################################################
SYSTEM = "You are a culinary assistant producing structured markdown."
MEAL_TEMPLATE = (
    "Return a 7‑day meal plan:\n\n| Day | Breakfast | Lunch | Dinner | Calories |\n|-----|-----------|-------|--------|----------|\n" + "\n".join([f"| {d} | | | | |" for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]))

def chat(msgs, temp=0.6):
    return client.chat.completions.create(model="gpt-3.5-turbo", messages=msgs, temperature=temp).choices[0].message.content.strip()

###############################################################################
# 📦 YOLOv7 Calorie Estimator - Embedded
###############################################################################
model = attempt_load("best.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()
class_names = model.names
nutritional_data = {
    "egg": {"calories": 68}, "rice": {"calories": 130}, "roti": {"calories": 71},
    "idli": {"calories": 58}, "dal": {"calories": 120}, "salad": {"calories": 35},
    "vada": {"calories": 132}, "curd": {"calories": 98}, "omelette": {"calories": 154},
}

def detect_and_overlay_nutrition_streamlit(image, conf_threshold=0.25, iou_threshold=0.45):
    img = np.array(image)
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized)).float().div(255.0).unsqueeze(0)

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, agnostic=False)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                cls_name = class_names[int(cls)]
                nutrition = nutritional_data.get(cls_name, None)
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                if nutrition:
                    label = f"{cls_name}: {nutrition['calories']} kcal"
                    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img

###############################################################################
# 📂 Navigation
###############################################################################
with st.sidebar:
    st.markdown("---")
    section = st.radio("Feature", ("Recipes","Meal Plans","Text Calories","Tracker","BMI Calculator","YOLO Calorie"))
    books   = st.multiselect("Cookbooks", list(COOKBOOKS), default=list(COOKBOOKS)) if section in {"Recipes","Meal Plans"} else list(COOKBOOKS)

###############################################################################
# 🖥️  Main interface
###############################################################################
if section in {"Recipes","Meal Plans"}:
    key = "prompt"; st.text_area("Your request", key=key, height=120)
    if st.button("📚 Add RAG Context"):
        ctx = rag.ctx(st.session_state[key], [COOKBOOKS[b] for b in books])
        st.session_state[key] = textwrap.dedent(f"{st.session_state[key]}\n\n# RAG Context\n{ctx}")
        st.success("Context added.")
    if st.button("🚀 Submit") and st.session_state.get(key):
        msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":st.session_state[key]}]
        out  = chat(msgs,0.7) if section=="Recipes" else chat([{ "role":"system","content":MEAL_TEMPLATE }]+msgs,0.4)
        st.markdown(out)
        if section=="Meal Plans":
            st.download_button("Download MD", out, "mealplan.md", "text/markdown")
            if st.button("🧾 Shopping List"):
                dishes = re.findall(r"\|\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|", out)
                flat   = [d for row in dishes for d in row if d.strip()]
                df     = pd.DataFrame({"dish": flat}); st.dataframe(df)
                st.download_button("CSV", df.to_csv(index=False).encode(), "shopping.csv", "text/csv")

elif section == "Text Calories":
    ing = st.text_area("Ingredients list (one per line)")
    if st.button("Estimate") and ing.strip():
        res = chat([{"role":"system","content":"List kcal for each ingredient then total."},{"role":"user","content":ing}],0.3)
        st.markdown(res)

elif section == "BMI Calculator":
    st.title('Welcome to BMI Calculator')
    weight = st.number_input('Enter your weight in kgs')
    status = st.radio('Select your height format:', ('cms','meters','feet'))
    try:
        if status == 'cms':
            height = st.number_input('Height in centimeters')
            bmi = weight / ((height / 100) ** 2)
        elif status == 'meters':
            height = st.number_input('Height in meters')
            bmi = weight / (height ** 2)
        elif status == 'feet':
            height = st.number_input('Height in feet')
            bmi = weight / ((height / 3.28) ** 2)
    except ZeroDivisionError:
        st.error("Height can't be zero!")

    if st.button('Calculate BMI'):
        st.write(f'Your BMI index is **{round(bmi, 2)}**.')
        if bmi < 16:
            st.error('You are extremely underweight')
        elif bmi < 18.5:
            st.warning('You are underweight')
        elif bmi < 25:
            st.success('You are healthy')
        elif bmi < 30:
            st.warning('You are overweight')
        else:
            st.error('You are extremely overweight')
        st.balloons()

elif section == "YOLO Calorie":
    st.subheader("YOLOv7 Food Calorie Estimator")
    st.markdown("Upload a food photo and detect calorie content using your trained YOLOv7 model.")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original", use_column_width=True)
        result = detect_and_overlay_nutrition_streamlit(image, conf_threshold=conf, iou_threshold=iou)
        st.image(result, caption="Detected", use_column_width=True)


