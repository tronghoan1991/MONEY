# === PHẦN 1: IMPORT, CẤU HÌNH, DB, FLASK, TIMEBLOCK ===
import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from datetime import datetime
import re
import joblib
import threading
import time
import warnings
import traceback
import io
import numpy as np
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ==== CẤU HÌNH ====
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

ROLLING_WINDOW = 12
MIN_BATCH = 5
TRAIN_EVERY = 5

MODEL_PATH = "ml_stack.joblib"
SESSION_FILE = "session_time.txt"
LAST_PLAY_FILE = "last_play_time.txt"

MIN_SESSION_INPUT = 10
SESSION_BREAK_MINUTES = 30
OVERRIDE_CUTOFF = 0.70

MARKOV_WINDOW = 100
STREAK_THRESHOLD = 5

# ==== FLASK KEEP-ALIVE ====
def start_flask():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Bot is alive!", 200

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ==== KHỞI TẠO DATABASE ====
def create_table():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE history ADD COLUMN input_time FLOAT;")
        conn.commit()
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            input TEXT,
            actual TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            bot_predict TEXT,
            input_time FLOAT DEFAULT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

# ==== XÁC ĐỊNH KHUNG GIỜ ====
def get_time_block():
    hour = datetime.now().hour
    if 5 <= hour <= 11:
        return 'sang'
    elif 12 <= hour <= 17:
        return 'chieu'
    else:
        return 'toi'

# === PHẦN 2: GHI LỊCH SỬ, TẠO FEATURES, PHIÊN CHƠI ===
def insert_result(input_str, actual, bot_predict=None, input_time=None):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    now = datetime.now()
    if bot_predict is not None:
        cur.execute("""
            INSERT INTO history (input, actual, bot_predict, created_at, input_time)
            VALUES (%s, %s, %s, %s, %s);
        """, (input_str, actual, bot_predict, now, input_time))
    else:
        cur.execute("""
            INSERT INTO history (input, actual, created_at, input_time)
            VALUES (%s, %s, %s, %s);
        """, (input_str, actual, now, input_time))
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=10000):
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM history ORDER BY id ASC LIMIT %s" % limit, engine)
    engine.dispose()
    return df[df["input"].str.match(r"^\d+$|^\d+\s+\d+\s+\d+$", na=False) | (df["input"] == "BOT_PREDICT")]

def summary_stats(df):
    df_pred = df[(df["bot_predict"].notnull()) & (df["actual"].notnull())]
    correct = (df_pred["bot_predict"] == df_pred["actual"]).sum()
    total = len(df_pred)
    wrong = total - correct
    acc = round((correct / total) * 100, 2) if total else 0
    return total, correct, wrong, acc

def save_session_start(time=None):
    with open(SESSION_FILE, "w") as f:
        f.write(str(time if time else datetime.now().isoformat()))

def load_session_start():
    if not os.path.exists(SESSION_FILE):
        return None
    with open(SESSION_FILE, "r") as f:
        try:
            return datetime.fromisoformat(f.read().strip())
        except:
            return None

def save_last_play(time=None):
    with open(LAST_PLAY_FILE, "w") as f:
        f.write(str(time if time else datetime.now().isoformat()))

def load_last_play():
    if not os.path.exists(LAST_PLAY_FILE):
        return None
    with open(LAST_PLAY_FILE, "r") as f:
        try:
            return datetime.fromisoformat(f.read().strip())
        except:
            return None

def make_features(df):
    df = df[df["input"].str.match(r"^\d+$|^\d+\s+\d+\s+\d+$", na=False)].copy()
    def extract_numbers(x):
        if " " in x:
            return list(map(int, x.split()))
        elif len(x) == 3 and x.isdigit():
            return [int(x[0]), int(x[1]), int(x[2])]
        return [0, 0, 0]

    df[["n1", "n2", "n3"]] = df["input"].apply(lambda x: pd.Series(extract_numbers(x)))
    df["total"] = df[["n1", "n2", "n3"]].sum(axis=1)
    df["even"] = df["total"] % 2
    df["bao"] = df.apply(lambda row: 1 if row["n1"] == row["n2"] == row["n3"] else 0, axis=1)
    df["tai"] = (df["total"] >= 11).astype(int)
    df["xiu"] = (df["total"] <= 10).astype(int)
    df["chan"] = (df["even"] == 0).astype(int)
    df["le"] = (df["even"] == 1).astype(int)

    for col in ["tai", "xiu", "chan", "le", "bao"]:
        df[f"{col}_roll"] = df[col].rolling(ROLLING_WINDOW, min_periods=1).mean()

    for i in range(1, 4):
        df[f"tai_lag_{i}"] = df["tai"].shift(i)
        df[f"chan_lag_{i}"] = df["chan"].shift(i)

    def get_streak(arr):
        streaks = [1]
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]:
                streaks.append(streaks[-1] + 1)
            else:
                streaks.append(1)
        return streaks

    df["tai_streak"] = get_streak(df["tai"].tolist())
    df["chan_streak"] = get_streak(df["chan"].tolist())
    return df

# === PHẦN 3: HUẤN LUYỆN VÀ LOAD MODEL THEO GIỜ ===
def train_models_by_timeblock(df):
    block = get_time_block()
    path = f"models_{block}.joblib"
    df = df.tail(ROLLING_WINDOW * 10)

    features = [
        'total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
        'tai_lag_1', 'tai_lag_2', 'tai_lag_3',
        'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
        'tai_streak', 'chan_streak'
    ]
    X = df[features].fillna(0)
    y_tx = df['tai']
    y_cl = df['chan']
    y_bao = df['bao']

    models = {}
    for key, y in [('tx', y_tx), ('cl', y_cl), ('bao', y_bao)]:
        if len(set(y)) < 2:
            models[key] = None
            continue
        lr = LogisticRegression().fit(X, y)
        rf = RandomForestClassifier(n_estimators=100).fit(X, y)
        xgbc = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss').fit(X, y)
        models[key] = (lr, rf, xgbc)

    joblib.dump(models, path)
    return path

def load_models_by_timeblock():
    block = get_time_block()
    path = f"models_{block}.joblib"
    return joblib.load(path) if os.path.exists(path) else None

# === PHẦN 4: BỔ SUNG HÀM THIẾU (SỬA LỖI) ===

def predict_stacking(X_pred, models, key):
    """
    Trả về xác suất dự đoán stacking trung bình của 3 model (LR, RF, XGB) với nhãn key (tx, cl, bao).
    """
    if models is None or key not in models or models[key] is None:
        return 0.5, None
    lr, rf, xgbc = models[key]
    probas = []
    for model in (lr, rf, xgbc):
        try:
            proba = model.predict_proba(X_pred)[:, 1][0]
        except Exception:
            proba = 0.5
        probas.append(proba)
    avg_proba = float(np.mean(probas))
    return avg_proba, probas

def get_confidence_label(proba):
    if proba >= 0.75 or proba <= 0.25:
        return "Cao"
    elif proba >= 0.6 or proba <= 0.4:
        return "Trung bình"
    else:
        return "Thấp"

def compute_markov_transition(df):
    return None

# === PHẦN 5: TELEGRAM HANDLERS – NHẬP PHIÊN, RESET ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gửi 3 số phiên gần nhất (ví dụ: 354) để BOT phân tích.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    await update.message.reply_text("Đã reset phiên chơi!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input_str = update.message.text.strip()
    if re.match(r"^\d{3}$", input_str):
        numbers = [int(x) for x in input_str]
    elif re.match(r"^\d+\s+\d+\s+\d+$", input_str):
        numbers = list(map(int, input_str.split()))
    else:
        await update.message.reply_text("❌ Vui lòng nhập đúng định dạng: 3 chữ số liền nhau (VD: 354)")
        return

    total = sum(numbers)
    actual = "Tài" if total >= 11 else "Xỉu"

    now = datetime.now()
    input_time = time.time()
    last_play = load_last_play()
    if last_play:
        delta = (now - last_play).total_seconds()
        if delta > SESSION_BREAK_MINUTES * 60:
            save_session_start(now)
    save_last_play(now)

    insert_result(" ".join(map(str, numbers)), actual, None, input_time)
    await predict(update, context)

# === PHẦN 6: DỰ ĐOÁN & PHẢN HỒI TIẾNG VIỆT ===

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    session_start = load_session_start()
    df_session = df[df['created_at'] >= session_start] if session_start else df

    if len(df_session[df_session['input'] != "BOT_PREDICT"]) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"⚠️ Cần tối thiểu {MIN_SESSION_INPUT} phiên để dự đoán.")
        return

    df_feat = make_features(df)
    df_feat_session = make_features(df_session)
    models = load_models_by_timeblock()

    features = ['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
                'tai_lag_1', 'tai_lag_2', 'tai_lag_3',
                'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
                'tai_streak', 'chan_streak']
    X_pred = df_feat_session.iloc[[-1]][features].fillna(0)

    tx_proba, _ = predict_stacking(X_pred, models, 'tx')
    cl_proba, _ = predict_stacking(X_pred, models, 'cl')

    decision = "Tài" if tx_proba >= 0.5 else "Xỉu"
    total_value = int(X_pred["total"].values[0])
    confidence = get_confidence_label(tx_proba)
    bao = int(X_pred["bao_roll"].values[0] > 0.2)
    streak = int(X_pred["tai_streak"].values[0])

    override_reason = None
    markov_info = compute_markov_transition(df_feat_session)
    if markov_info and abs(tx_proba - 0.5) <= (1 - OVERRIDE_CUTOFF):
        last = markov_info['last']
        if last == "T" and decision == "Tài" and markov_info['prob_T2X'] > markov_info['prob_X2T']:
            decision = "Xỉu"
            override_reason = "Markov cho thấy xác suất đảo chiều cao"
        elif last == "X" and decision == "Xỉu" and markov_info['prob_X2T'] > markov_info['prob_T2X']:
            decision = "Tài"
            override_reason = "Markov cho thấy xác suất đảo chiều cao"

    if not override_reason and streak >= STREAK_THRESHOLD and abs(tx_proba - 0.5) <= (1 - OVERRIDE_CUTOFF):
        decision = "Xỉu" if decision == "Tài" else "Tài"
        override_reason = f"Chuỗi {('Tài' if decision == 'Xỉu' else 'Xỉu')} liên tiếp ({streak} phiên)"

    insert_result("BOT_PREDICT", None, decision)

    total, correct, wrong, acc = summary_stats(df)
    result_msg = [f"BOT dự đoán phiên tiếp theo: {decision}"]
    result_msg.append(f"Dải điểm nên đánh: {'11 → 17' if decision == 'Tài' else '4 → 10'}")
    if override_reason:
        result_msg.insert(1, f"Lý do: {override_reason}")
    if confidence == "Thấp":
        result_msg.append("*Lưu ý: Xác suất hiện tại không rõ ràng, cân nhắc không vào*")
    if bao:
        result_msg.append("⚡ Có thể vào bão")
    result_msg.append(f"Tổng phiên dự đoán: {total} | Đúng: {correct} | Sai: {wrong} | Chính xác: {acc}%")

    await update.message.reply_text("\n".join(result_msg))

# === PHẦN CUỐI: KHỞI CHẠY FLASK + POLLING ===

if __name__ == "__main__":
    import asyncio
    create_table()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Khởi chạy Flask keep-alive song song với Telegram polling
    threading.Thread(target=start_flask, daemon=True).start()

    asyncio.run(app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False, stop_signals=None))
