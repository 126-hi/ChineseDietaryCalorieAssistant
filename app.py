# -*- coding: utf-8 -*-
"""
Streamlit Multi-Agent Cuisine Assistant – Fully-Integrated Final Build (美化增强版)
==============================================================================
"""

###############################################################################
# 🔧 Imports & Path
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
# 💌 Global pastel theme + Button Style
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
""",unsafe_allow_html=True)

###############################################################################
# 🌟 Top Logo & Title
###############################################################################
st.markdown("""
<h1 style='text-align: center;'>🥢 Chinese Cuisine & Nutrition Assistant</h1>
<p style='text-align: center; color: grey;'>Your AI-powered Chinese Meal Planner and Tracker</p>
""", unsafe_allow_html=True)

###############################################################################
# 🔑 API Keys (sidebar)
###############################################################################
with st.sidebar:
    st.title("API Keys")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

if not api_key:
    st.warning("\u2139\ufe0f Enter your OpenAI key to continue …")
    st.stop()

openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

###############################################################################
# 📓 Persisted system prompt for chat
###############################################################################
SYSTEM_PROMPT = (
    "You are a culinary assistant who helps users create authentic Chinese recipes based on available ingredients.\n"
    "Provide a structured response with dish name (EN & CN), ingredients list, step-by-step instructions, and cooking tips.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":SYSTEM_PROMPT}]

###############################################################################
# 📓 Cookbooks & FAISS retriever
###############################################################################
COOKBOOKS = {
    "Easy Chinese Cuisine":"data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "China in 50 Dishes":"data/02. China in 50 Dishes author HSBC.pdf",
    "7-Day Healthy Meal Plan":"data/7-day-Chinese-healthy-meal-plan.pdf",
}

@st.cache_resource(show_spinner="Indexing cookbooks …")
def build_retriever(paths:List[str]):
    docs = []
    for p in paths:
        if Path(p).exists():
            docs += PyPDFLoader(p).load()
    splits = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    store = FAISS.from_documents(splits, OpenAIEmbeddings(openai_api_key=api_key))
    return store.as_retriever()

retriever = build_retriever(list(COOKBOOKS.values()))

###############################################################################
# 🤖 Chat helpers
###############################################################################
def chat(msgs, temp=0.6):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=msgs,
        temperature=temp
    ).choices[0].message.content.strip()

def chat_with_cookbook(user_query, k=3, temp=0.6):
    related_docs = retriever.get_relevant_documents(user_query, k=k)
    context = "\n\n".join(doc.page_content for doc in related_docs)
    full_prompt = (
        f"You are a Chinese cuisine assistant. Based on the following cookbook references:\n\n"
        f"{context}\n\n"
        f"And the user's request:\n{user_query}\n\n"
        f"Generate a detailed and authentic Chinese-style recipe or meal plan.")
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":full_prompt}],
        temperature=temp
    ).choices[0].message.content.strip()

###############################################################################
# 🛀 Sidebar navigation (emoji enhanced)
###############################################################################
with st.sidebar:
    st.markdown("---")
    section = st.radio("Feature", ("🏠 Home", "🍽️ Recipes", "📅 Meal Plan", "🗓️ Calendar", "⚖️ BMI", "📷 YOLO", "🥗 Nutrition"))



###############################################################################
# 🔌 YOLOv7 Loader & Infer
###############################################################################
@st.cache_resource(show_spinner="Loading YOLOv7 …")
def load_model(path="best.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = attempt_load(path, map_location=device)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()
    model.to(device).eval()
    return model, device

model, device = load_model()
class_names = model.names
nutri = {"egg": 68, "rice": 130, "salad": 35}

# Preprocess and detect function
def preprocess(image):
    arr = np.array(image)
    r = letterbox(arr, 640)[0]
    r = r[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(r)).float().div(255).unsqueeze(0), arr

def detect(image, conf=0.25, iou=0.45):
    t, a = preprocess(image)
    t = t.to(device)
    with torch.no_grad():
        pred = non_max_suppression(model(t)[0], conf, iou)[0]
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(t.shape[2:], pred[:, :4], a.shape).round()
        for *xy, conf_score, cls in pred:
            name = class_names[int(cls)]
            kcal = nutri.get(name, '?')
            cv2.rectangle(a, (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3])), (0, 255, 0), 2)
            cv2.putText(a, f"{name}: {kcal}kcal", (int(xy[0]), int(xy[1]) - 6), 0, 0.5, (0, 0, 0), 2)
    return a

###############################################################################
# 🏠 Home – Welcome Only
###############################################################################
if section == "🏠 Home":
    st.header("🏠 Welcome!")
    st.write("Use the sidebar to explore Recipes, Meal Plans, Nutrition Tracking, and more!")

###############################################################################
# 🍽️ Recipes / 📅 Meal Plan – RAG Enhanced & Unified Calendar
###############################################################################
if section == "🍽️ Recipes":
    st.header("🍽️ Recipe Generator")
    prompt = st.text_area("Describe the dish you want:", key="recipe_prompt", height=120)
    if st.button("Generate Recipe"):
        reply = chat_with_cookbook(prompt)
        st.markdown(reply)

if section == "📅 Meal Plan":
    st.header("📅 Meal Plan & Calendar")

    # 🔹 Input fields
    ingredients = st.text_input("Ingredients available:", value="tofu, beef, broccoli, garlic, egg")
    calorie_target = st.number_input("Target Calories per Day:", min_value=500, max_value=4000, value=1500)

    if st.button("Generate Meal Plan"):
        user_request = (
            f"Create a 7-day Chinese meal plan using: {ingredients}. "
            f"Each day should be around {calorie_target} calories. "
            "Format output as a markdown table: Day | Dish | Ingredients | Estimated Calories."
        )
        reply = chat_with_cookbook(user_request)
        st.session_state.mealplan_md = reply

        # 🔹 Parse structured meal plan for Calendar
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        mealplan_list = []
        for day in days:
            pattern = fr"{day}\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)"
            match = re.search(pattern, reply)
            if match:
                dish = match.group(1).strip()
                calories = match.group(3).strip()
                mealplan_list.append({"day": day, "dish": dish, "calories": calories})

        st.session_state.mealplan_list = mealplan_list

    # 🔹 Display Meal Plan on Calendar
    if "mealplan_list" in st.session_state:
        today = dt.date.today()
        ev = []
        for idx, item in enumerate(st.session_state.mealplan_list):
            day_date = today + dt.timedelta(days=idx)
            title = f"{item['dish']} ({item['calories']})"
            ev.append({"title": title, "start": day_date.isoformat()})

        st.markdown("### 🗓️ Meal Plan Calendar")
        calendar(events=ev, options={"initialView": "dayGridWeek"})
    else:
        st.info("No meal plan detected. Generate above!")

###############################################################################
# ⚖️ BMI Calculator
###############################################################################
if section == "⚖️ BMI":
    w = st.number_input("Weight (kg)", 0.0, 300.0, 70.0)
    h = st.number_input("Height (cm)", 0.0, 250.0, 170.0)
    if st.button("Calculate BMI") and h > 0:
        bmi = w / ((h/100)**2)
        st.success(f"BMI = {bmi:.1f}")

###############################################################################
# 📷 YOLO Food Calorie Estimation
###############################################################################
if section == "📷 YOLO":
    up = st.file_uploader("Upload food image:", type=["jpg", "png"])
    if up:
        arr = detect(Image.open(up))
        st.image(Image.fromarray(arr), use_column_width=True)

###############################################################################
# 🥗 Nutrition Tracker (with Emoji)
###############################################################################
if section == "🥗 Nutrition":
    st.header("🥗 Food Nutrition Tracker")
    today = dt.date.today().isoformat()

    if "food_log" not in st.session_state:
        st.session_state.food_log = {}
    if today not in st.session_state.food_log:
        st.session_state.food_log[today] = []

    food_db = {
        "egg": {"calories": 68, "protein": 6, "carbs": 1, "fat": 5},
        "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
        "salad": {"calories": 35, "protein": 2, "carbs": 7, "fat": 0.2},
        "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3},
    }
    emoji_map = {"egg": "🍳", "salad": "🥗", "rice": "🍚", "chicken breast": "🍗", "apple": "🍎"}

    food = st.selectbox("Select a food", list(food_db.keys()))
    qty = st.number_input("Quantity (servings)", 1, 10, 1)

    if st.button("Add Food"):
        entry = food_db[food].copy()
        entry["name"] = emoji_map.get(food, "") + " " + food
        entry["quantity"] = qty
        st.session_state.food_log[today].append(entry)

    if st.session_state.food_log[today]:
        df = pd.DataFrame(st.session_state.food_log[today])
        df["Total Calories"] = df["calories"] * df["quantity"]
        df["Total Protein"] = df["protein"] * df["quantity"]
        df["Total Carbs"] = df["carbs"] * df["quantity"]
        df["Total Fat"] = df["fat"] * df["quantity"]

        st.dataframe(df[["name", "quantity", "Total Calories", "Total Protein", "Total Carbs", "Total Fat"]])

        total_cal = df["Total Calories"].sum()
        total_pro = df["Total Protein"].sum()
        total_carb = df["Total Carbs"].sum()
        total_fat = df["Total Fat"].sum()

        st.success(f"Today Total: {total_cal:.0f} kcal | Protein {total_pro:.1f}g | Carbs {total_carb:.1f}g | Fat {total_fat:.1f}g")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download today's log as CSV", data=csv, file_name=f"nutrition_log_{today}.csv", mime='text/csv')
    else:
        st.info("No foods added yet for today. Start by adding!")


