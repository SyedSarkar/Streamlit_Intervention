import json
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from transformers.pipelines import pipeline
import string
import nltk
from nltk.corpus import words
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
import datetime

# ---- Google Sheets Setup ----
HEADERS = ["timestamp", "user", "phase", "cue", "sentence", "response", "sentiment",
           "confidence", "score", "response_time_sec", "accepted"]

@st.cache_resource
def connect_to_sheet():
    creds_json = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])
    creds = Credentials.from_service_account_info(
        creds_json,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"   # ✅ Add this
        ]
    )
    client = gspread.authorize(creds)
    files = client.list_spreadsheet_files()  # Now this will work
    #st.write("Sheets accessible:", files)
    return client.open("Intervention_Results").sheet1

def log_to_gsheet(row_dict):
    sheet = connect_to_sheet()
    row = [str(row_dict.get(col, "")) for col in HEADERS]
    sheet.append_row(row)
# ----------------------------------

# Download NLTK words if not already downloaded
nltk.download('words')
english_vocab = set(w.lower() for w in words.words())

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'on', 'in', 'with', 'to', 'from', 'by',
    'of', 'for', 'at', 'as', 'is', 'it', 'this', 'that', 'these', 'those', 'i', 'you',
    'he', 'she', 'we', 'they', 'them', 'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'be', 'am', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
}

def looks_like_gibberish(word):
    return (
        len(word) < 2 or
        not word.isalpha() or
        re.fullmatch(r"(.)\1{2,}", word) or
        re.search(r'[aeiou]{3,}', word) or
        re.search(r'[zxcvbnm]{4,}', word) or
        word not in english_vocab
    )

def is_valid_response(response, cue_word):
    tokens = response.lower().strip().split()
    if not 1 <= len(tokens) <= 3:
        return False
    for token in tokens:
        if token == cue_word.lower() or token in STOPWORDS or looks_like_gibberish(token):
            return False
    return True

def calculate_score(label):
    if label == "POSITIVE": return 2
    if label == "NEGATIVE": return -1
    return 1

def format_cue_word(cue):
    return f"""
    <div style='text-align: center; font-size: 32px; font-weight: bold; color: #010d1a; padding: 20px;'>
        {cue}
    </div>
    """

def format_feedback(msg, color):
    return f"""
    <div style='text-align: center; font-size: 24px; font-weight: bold; color: {color}; padding: 10px;'>
        {msg}
    </div>
    """


def get_safe_progress(current, total):
    if total == 0:
        return 0.0
    return min(max(current / total, 0.0), 1.0)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

os.makedirs("results", exist_ok=True)

with open("data/cue_words.txt", "r") as f:
    cue_words = [line.strip() for line in f.readlines()]

with open("data/sentences.txt", "r") as f:
    sentences = [line.strip() for line in f.readlines()]

if "phase" not in st.session_state:
    st.session_state.user_id = ""
    st.session_state.phase = 0
    st.session_state.step = 0
    st.session_state.score = 0
    st.session_state.used_texts = set()
    st.session_state.responses = []
    st.session_state.start_time = None

st.markdown("""
<style>
body {
    background-color: #f6f9fc;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.phase == 0:
    st.title("Positive Phrase Intervention")
    st.markdown("""
    Welcome to this two-phase task designed to encourage positive associations and emotional reflection.

    - **Phase 1**: Respond to single cue words with uplifting phrases.
    - **Phase 2**: React to full sentences with encouraging responses.
    - Avoid repeats and generic prepositions.
    """)
    user_input = st.text_input("Enter your Name or Roll Number:")
    if st.button("Start Task") and user_input.strip():
        st.session_state.user_id = user_input.strip()
        safe_id = re.sub(r'[^\w\-]', '_', user_input.strip())
        filename = f"results/{safe_id}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            st.session_state.responses = df.to_dict("records")
            st.session_state.used_texts = set(df["response"].dropna().str.lower().tolist())
            st.session_state.score = df["score"].sum()
            st.session_state.step = sum(1 for r in st.session_state.responses if r["phase"] == 1)
            st.session_state.phase = 2 if st.session_state.step >= len(cue_words) else 1
        else:
            st.session_state.phase = 1
        st.rerun()

if st.session_state.phase == 1:
    st.progress(get_safe_progress(st.session_state.step, len(cue_words)))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")

    if st.session_state.step < len(cue_words):
        cue = cue_words[st.session_state.step]
        st.markdown(format_cue_word(cue), unsafe_allow_html=True)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input():
            phrase = st.session_state[f"input_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            entry = {
                "timestamp": str(datetime.datetime.now()),
                "user": st.session_state.user_id,
                "phase": 1,
                "cue": cue,
                "sentence": "",
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            if not is_valid_response(phrase, cue):
                feedback.markdown(format_feedback("❌ Invalid input!", "red"), unsafe_allow_html=True)
                time.sleep(2)
            elif phrase in st.session_state.used_texts:
                feedback.markdown(format_feedback("⚠️ Already used!", "orange"), unsafe_allow_html=True)
                time.sleep(2)
            elif label == "NEGATIVE":
                feedback.markdown(format_feedback("❌ Negative! Try again.", "red"), unsafe_allow_html=True)
                time.sleep(2)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "green"), unsafe_allow_html=True)
                time.sleep(2)

            st.session_state.responses.append(entry)
            safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
            pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
            log_to_gsheet(entry)  # <-- Logging to Google Sheet

        st.text_input("Type a related uplifting phrase (up to 3 words):", key=f"input_{st.session_state.step}", on_change=handle_input)

    else:
        st.success("🎉 Congratulations Phase 1 Complete!")
        if st.button("Proceed to Phase 2"):
            st.session_state.step = 0
            st.session_state.phase = 2
            st.rerun()

elif st.session_state.phase == 2:
    st.progress(get_safe_progress(st.session_state.step, len(sentences)))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")

    if st.session_state.step < len(sentences):
        sentence = sentences[st.session_state.step]
        st.subheader(f"Sentence {st.session_state.step + 1}:")
        st.write(f"**{sentence}**")

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input_2():
            phrase = st.session_state[f"input_s2_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            entry = {
                "timestamp": str(datetime.datetime.now()),
                "user": st.session_state.user_id,
                "phase": 2,
                "cue": "",
                "sentence": sentence,
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            if not is_valid_response(phrase, sentence):
                feedback.markdown(format_feedback("❌ Invalid input!", "red"), unsafe_allow_html=True)
                time.sleep(2)
            elif phrase in st.session_state.used_texts:
                feedback.markdown(format_feedback("⚠️ Already used!", "orange"), unsafe_allow_html=True)
                time.sleep(2)
            elif label == "NEGATIVE":
                feedback.markdown(format_feedback("❌ Negative! Try again.", "red"), unsafe_allow_html=True)
                time.sleep(2)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "green"), unsafe_allow_html=True)
                time.sleep(2)

            st.session_state.responses.append(entry)
            safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
            pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
            log_to_gsheet(entry)  # <-- Logging to Google Sheet

        st.text_input("Respond with a positive phrase:", key=f"input_s2_{st.session_state.step}", on_change=handle_input_2)

    else:
        st.session_state.step = 0
        st.session_state.phase = 3
        st.rerun()

elif st.session_state.phase == 3:
    st.balloons()
    st.success("🎉 Congratulations on Completing the Task!")
    st.markdown(f"**Final Score:** `{st.session_state.score}`")
    df = pd.DataFrame(st.session_state.responses)
    st.dataframe(df)

    with st.expander("📊 Analytics Dashboard"):
        st.subheader("Confidence Over Time")
        df["step"] = range(1, len(df) + 1)
        min_step, max_step = st.slider("Select step range:", int(df["step"].min()), int(df["step"].max()), (int(df["step"].min()), int(df["step"].max())))
        filtered_df = df[(df["step"] >= min_step) & (df["step"] <= max_step)]

        fig1, ax1 = plt.subplots()
        filtered_df.plot(x="step", y="confidence", ax=ax1, color="green", marker='o')
        st.pyplot(fig1)

        st.subheader("Score Over Time")
        filtered_df["cumulative"] = filtered_df["score"].cumsum()
        fig2, ax2 = plt.subplots()
        filtered_df.plot(x="step", y="cumulative", ax=ax2, color="blue", marker='o')
        st.pyplot(fig2)

    st.download_button("Download Results", df.to_csv(index=False).encode(), file_name=f"{st.session_state.user_id}_results.csv")

    if st.button("🔁 Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
