# -*- coding: utf-8 -*-
"""
Streamlit Multi‑Agent Cuisine Assistant – Full Version
======================================================
This script merges the **original demo** with the **new capability blocks**:

1. 💡 Personalised goal calculator (BMR/TDEE) – sidebar
2. 🧑‍🍳 Recipes with RAG + diet filters & preferences
3. 🍱 Weekly meal‑plan generator with export + shopping list
4. 📷 Food image generation (DALL·E 3)
5. 🖼️ Photo‑based calorie estimation (HF Calorie_counter)
6. 🧾 Ingredient‑list calorie estimation (GPT‑3.5)
7. 📈 Daily calorie & weight tracker (Plotly)

**How to run**
```
streamlit run cuisine_assistant_app.py
```
Fill in OpenAI & HF tokens in the sidebar.
"""

from __future__ import annotations
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base64
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

###############################################################################
# 🎨  Global styling & page config
###############################################################################
CSS = """
<style>
  .stApp         { background-color:#fff9e6; }
  .title         { text-align:center; font-size:36px; font-weight:bold; }
  .stChatMessage { border-radius:10px; padding:10px; margin-bottom:5px; }
  .stTextInput>div>div>input { border-radius:10px; border:1px solid #FFA500; }
  .css-1d391kg   { background-color:#fff3d4 !important; }
</style>
"""

st.set_page_config(page_title="Cuisine Assistant", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("<h1 class='title'>🍽️ Multi‑Agent Cuisine Assistant</h1>", unsafe_allow_html=True)

###############################################################################
# 🔑  API keys & personal goal (SIDEBAR)
###############################################################################
with st.sidebar:
    st.title("🔑 API Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password")
    hf_token_input = st.text_input("Hugging Face Access Token", type="password")

# Personal Goal Calculator ----------------------------------------------------
_ACTIVITY_FACTOR = {
    "Sedentary (little / no exercise)": 1.2,
    "Light (1–3 days/wk)": 1.375,
    "Moderate (3–5 days/wk)": 1.55,
    "Active (6–7 days/wk)": 1.725,
    "Very Active (physical job)": 1.9,
}
_GOAL_ADJUST = {
    "Lose weight": -500,
    "Maintain": 0,
    "Gain muscle": 300,
}

def mifflin_st_jeor(sex: str, w: float, h: float, age: int) -> float:
    return (10*w + 6.25*h - 5*age + (5 if sex=="Male" else -161))

with st.sidebar:
    st.markdown("---")
    st.header("🎯 Personal Goal (BMR/TDEE)")
    sex = st.radio("Sex", ("Male", "Female"), horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 10, 100, value=25)
        weight = st.number_input("Weight (kg)", 30.0, 300.0, value=70.0, format="%0.1f")
    with col2:
        height = st.number_input("Height (cm)", 120.0, 230.0, value=170.0, format="%0.1f")
        goal_type = st.selectbox("Goal", list(_GOAL_ADJUST))
    activity = st.selectbox("Activity Level", list(_ACTIVITY_FACTOR))
    if st.button("⚡ Calculate"):
        bmr = mifflin_st_jeor(sex, weight, height, age)
        tdee = bmr * _ACTIVITY_FACTOR[activity]
        target = max(1000, round(tdee + _GOAL_ADJUST[goal_type]))
        st.success(f"Daily target ≈ **{target} kcal** (TDEE {tdee:.0f})")
        st.session_state["recommended_calories"] = target

###############################################################################
# 🚀  Initialise APIs
###############################################################################
if not api_key_input:
    st.warning("Please provide an OpenAI API key in the sidebar.")
    st.stop()
openai.api_key = api_key_input
client = openai.OpenAI(api_key=api_key_input)

###############################################################################
# 📚  Build / load FAISS retriever (cookbook PDFs)
###############################################################################
PDF_PATHS = [
    "/mnt/data/01. Easy Chinese Cuisine author Ailam Lim.pdf",
    "/mnt/data/02. China in 50 Dishes author HSBC.pdf",
]

@st.cache_resource(show_spinner="🔍 Building vector database…")
def load_retriever(paths: List[str], key: str):
    pages = []
    for p in paths:
        if Path(p).exists():
            pages.extend(PyPDFLoader(p).load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_pages = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(api_key=key)
    store = FAISS.from_documents(split_pages, embeddings)
    return store.as_retriever()

retriever = load_retriever(PDF_PATHS, api_key_input)

###############################################################################
# 🛠️  Agents
###############################################################################
SYSTEM_PROMPT = (
    "You are a culinary assistant who creates authentic Chinese recipes. "
    "Always return structured markdown."
)
class RAGAgent:
    def __init__(self, retriever):
        self.retriever = retriever
    def get_context(self, query: str, k: int = 3) -> str:
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join(d.page_content for d in docs[:k])
rag_agent = RAGAgent(retriever)

class RecipeAgent:
    def generate(self, prompt: str, prefs: str = "", use_rag: bool = True) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if use_rag:
            msgs.append({"role": "system", "content": f"Cookbook context:\n{rag_agent.get_context(prompt)}"})
        msgs.append({"role": "user", "content": prompt + ("\n"+prefs if prefs else "")})
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=msgs, temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
recipe_agent = RecipeAgent()

class MealPlanAgent:
    TEMPLATE = (
        "Create a 7‑day meal plan. Return markdown table with Day, Meal, Dish, Estimated Calories." )
    def generate(self, prompt: str, prefs: str = "") -> str:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.TEMPLATE},
                {"role": "user", "content": prompt + ("\n"+prefs if prefs else "")},
            ],
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()
meal_agent = MealPlanAgent()

class ImageAgent:
    def generate(self, prompt: str) -> str:
        resp = client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
        return resp.data[0].url
image_agent = ImageAgent()

class CalorieTextAgent:
    TEMPLATE = "Estimate total calories for the ingredients listed. Return each ingredient with kcal and a final sum."
    def estimate(self, ingredients: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": self.TEMPLATE}, {"role": "user", "content": ingredients}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
calorie_text_agent = CalorieTextAgent()

class CalorieVisionAgent:
    API_URL = "https://api-inference.huggingface.co/models/JaydeepR/Calorie_counter"
    def __init__(self, hf_token: Optional[str]):
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    def estimate(self, img_bytes: bytes) -> str:
        resp = requests.post(self.API_URL, headers=self.headers, data=img_bytes)
        if resp.status_code != 200:
            return f"❌ HF API error: {resp.status_code} – {resp.text}"
        try:
            data = resp.json()
        except ValueError:
            return "❌ Invalid JSON from HF API."
        return "\n".join([f"**{k.title()}**: {v}" for k, v in data.items()])
calorie_vision_agent = CalorieVisionAgent(hf_token_input)

###############################################################################
# 🔧  Diet preference helpers
###############################################################################

def preference_form() -> Dict[str, str | List[str]]:
    with st.expander("🛠️ Diet Filters & Preferences"):
        liked = st.text_input("👍 Preferred ingredients (comma‑separated)")
        disliked = st.text_input("👎 Avoid ingredients (comma‑separated)")
        diets = st.multiselect("Dietary style", ["Vegetarian", "Low‑carb", "High‑protein"])
    return {"liked": liked, "disliked": disliked, "diets": diets}

def prefs_to_str(pref: Dict[str, str | List[str]]) -> str:
    parts = []
    if pref.get("liked"):
        parts.append(f"Preferred: {pref['liked']}")
    if pref.get("disliked"):
        parts.append(f"Avoid: {pref['disliked']}")
    if pref.get("diets"):
        parts.append("Diet style: " + ", ".join(pref["diets"]))
    return "\n".join(parts)

###############################################################################
# 📂  Sidebar navigation
###############################################################################
with st.sidebar:
    st.markdown("---")
    st.title("📂 Navigation")
    section = st.radio(
        "Choose a feature:",
        (
            "Recipes",
            "Make Meal Plans",
            "Take Photos",
            "Photo Calorie Counter",
            "Calculate Calories (Text)",
            "Calorie Tracker",
        ),
    )
    if section == "Recipes":
        use_rag_flag = st.checkbox("✨ Use RAG (cookbook PDFs)", value=True)

###############################################################################
# 🖥️  Main interface
###############################################################################

def make_download_buttons(markdown_str: str, file_stub: str):
    st.download_button("⬇️ Download (Markdown)", markdown_str, f"{file_stub}.md", mime="text/markdown")
    try:
        import pdfkit  # noqa: F401
        pdf_bytes = pdfkit.from_string(markdown_str, False)
        st.download_button("⬇️ PDF", pdf_bytes, f"{file_stub}.pdf", mime="application/pdf")
    except Exception:
        pass

def extract_ingredients(md: str) -> pd.DataFrame:
    block_rgx = re.compile(r"\|\s*Meal\s*\|.+\n((?:\|.+\n)+)")
    lines = []
    m = block_rgx.search(md)
    if not m:
        return pd.DataFrame()
    for row in m.group(1).splitlines():
        cols = [c.strip() for c in row.split("|") if c.strip()]
        if len(cols) >= 3:
            lines.append(cols[2])
    return pd.DataFrame({"ingredient": lines})

def shopping_list_ui(md: str):
    st.subheader("🧾 Shopping List")
    df = extract_ingredients(md)
    if df.empty:
        st.info("No ingredient block detected.")
        return
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "shopping_list.csv", mime="text/csv")

# -------- Calorie Tracker Helpers -------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = pd.DataFrame(columns=["date", "meal", "calories", "weight", "target"])

def tracker_ui():
    st.header("📈 Daily Calorie & Weight Tracker")
    with st.form("log_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date", dt.date.today())
            meal = st.text_input("Meal / Dish")
        with col2:
            cal = st.number_input("Calories", 0, 3000, step=10)
            weight = st.number_input("Weight (kg)", 0.0, 300.0, step=0.1)
        submitted = st.form_submit_button("Add")
    if submitted:
        target = st.session_state.get("recommended_calories", None)
        st.session_state.log = pd.concat(
            [st.session_state.log, pd.DataFrame([[date, meal, cal, weight or None, target]], columns=st.session_state.log.columns)],
            ignore_index=True,
        )
        st.success("Entry added!")
    log_df = st.session_state.log.sort_values("date")
    if log_df.empty:
        st.info("No entries yet.")
        return
    st.dataframe(log_df, height=250, use_container_width=True)
    fig = px.line(log_df, x="date", y=["calories", "target"], markers=True, labels={"value": "Calories"})
    st.plotly_chart(fig, use_container_width=True)
    if log_df["weight"].notna().any():
        fig2 = px.line(log_df.dropna(subset=["weight"]), x="date", y="weight", markers=True, title="Weight trend (kg)")
        st.plotly_chart(fig2, use_container_width=True)

###############################################################################
# 🚦  Orchestration & UI Rendering
###############################################################################

if section == "Calorie Tracker":
    tracker_ui()
    st.stop()

prefs = preference_form() if section in {"Recipes", "Make Meal Plans"} else {}
prefs_str = prefs_to_str(prefs) if prefs else ""

if section == "Photo Calorie Counter":
    st.subheader("Upload a food photo (JPG/PNG)…")
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    if st.button("🚀 Estimate Calories"):
        if uploaded is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("🧮 Estimating…"):
                img_bytes = uploaded.read()
                result = calorie_vision_agent.estimate(img_bytes)
            st.image(img_bytes, caption="Uploaded", use_column_width=True)
            st.markdown(result)
else:
    placeholders = {
        "Recipes": "Enter ingredients or a dish name…",
        "Make Meal Plans": "Describe your dietary goals…",
        "Take Photos": "Describe the food image you want…",
        "Calculate Calories (Text)": "List ingredients and quantities…",
    }
    prompt = st.text_area(placeholders.get(section, "Your input…"), height=100)
    if st.button("🚀 Submit"):
        if not prompt.strip():
            st.warning("Please enter some text.")
            st.stop()
        with st.spinner("🤖 Working…"):
            if section == "Recipes":
                output = recipe_agent.generate(prompt.strip(), prefs_str, use_rag=use_rag_flag)
            elif section == "Make Meal Plans":
                output = meal_agent.generate(prompt.strip(), prefs_str)
            elif section == "Take Photos":
                output = image_agent.generate(prompt.strip())
            elif section == "Calculate Calories (Text)":
                output = calorie_text_agent.estimate(prompt.strip())
            else:
                output = "Unknown feature."
        if section == "Take Photos":
            st.image(output, caption="Generated", use_column_width=True)
        else:
            st.markdown(output)
            if section == "Make Meal Plans":
                make_download_buttons(output, "weekly_plan")
                if st.button("🧾 Generate shopping list"):
                    shopping_list_ui(output)

st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, FAISS, Hugging Face & OpenAI.*")

