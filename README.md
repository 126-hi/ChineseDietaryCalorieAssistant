# 🥢 Chinese Cuisine & Nutrition Assistant

> **AI-powered Chinese meal-planning, calorie tracking & food-recognition app  
> built with Streamlit, LangChain (RAG) & YOLOv7**

![hero](docs/hero.png)

---

## Features

| Module | What it does |
|--------|--------------|
| **Recipe Generator** | RAG-enhanced chat; generates authentic Chinese recipes (EN + CN names, ingredients, instructions) using 3 cook-books as context. |
| **7-Day Meal Plan**  | One-click table (Day \| Dish \| Ingredients \| Calories) adhering to a user-defined kcal target. |
| **BMI Calculator**   | Simple height/weight → BMI with colour feedback. |
| **YOLO Calorie Cam** | Upload a food image → YOLOv7 detects dish & overlays estimated kcal. |
| **Nutrition Tracker**| Emoji food picker; logs servings, totals calories / protein and exports CSV. |

---

## Demo

> *(drop a GIF or Streamlit Cloud link here)*

---

## Quick start

```bash
# 1️⃣ clone repo & create env
git clone https://github.com/<you>/cuisine-assistant.git
cd cuisine-assistant
python -m venv .venv && source .venv/bin/activate

# 2️⃣ install deps
pip install -r requirements.txt

# 3️⃣ download YOLOv7 weights (or use your own)
mkdir weights && wget -O weights/best.pt https://<your-link>/best.pt

# 4️⃣ export your OpenAI key
export OPENAI_API_KEY="sk-..."

# 5️⃣ run
streamlit run app.py
