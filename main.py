import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
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
PROBA_CUTOFF = 0.62
PROBA_ALERT = 0.75
BAO_CUTOFF = 0.03
TRAIN_EVERY = 5

MODEL_PATH = "ml_stack.joblib"
MODEL_META = "ml_meta.txt"
SESSION_FILE = "session_time.txt"
LAST_PLAY_FILE = "last_play_time.txt"

MIN_SESSION_INPUT = 10
SESSION_BREAK_MINUTES = 30
INPUT_DELAY_SEC = 90

MARKOV_WINDOW = 100
STREAK_THRESHOLD = 5

# ==== FLASK DÙNG CHO UPTIME ROBOT ====
def start_flask():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Bot is alive!", 200

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ==== TẠO BẢNG DB ====
def create_table():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE history ADD COLUMN input_time FLOAT;")
        conn.commit()
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            input TEXT,
            actual TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            bot_predict TEXT,
            input_time FLOAT DEFAULT NULL
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()
def insert_result(input_str, actual, bot_predict=None, input_time=None):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    now = datetime.now()
    if bot_predict is not None:
        cur.execute(
            "INSERT INTO history (input, actual, bot_predict, created_at, input_time) VALUES (%s, %s, %s, %s, %s);",
            (input_str, actual, bot_predict, now, input_time),
        )
    else:
        cur.execute(
            "INSERT INTO history (input, actual, created_at, input_time) VALUES (%s, %s, %s, %s);",
            (input_str, actual, now, input_time),
        )
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=10000):
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM history ORDER BY id ASC LIMIT %s" % limit, engine)
    engine.dispose()
    return df[df["input"].str.match(r"^\d+\s+\d+\s+\d+$", na=False) | (df["input"] == "BOT_PREDICT")]

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

# === TẠO ĐẶC TRƯNG (FEATURE ENGINEERING) ===
def make_features(df):
    df = df[df["input"].str.match(r"^\d+\s+\d+\s+\d+$", na=False)].copy()
    df["total"] = df["input"].apply(lambda x: sum([int(i) for i in x.split()]))
    df["even"] = df["total"] % 2
    df["bao"] = df["input"].apply(lambda x: 1 if len(set(x.split())) == 1 else 0)
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
def train_models(df):
    df = df.tail(ROLLING_WINDOW * 10)
    features = [
        "total", "even", "tai_roll", "xiu_roll", "chan_roll", "le_roll", "bao_roll",
        "tai_lag_1", "tai_lag_2", "tai_lag_3",
        "chan_lag_1", "chan_lag_2", "chan_lag_3",
        "tai_streak", "chan_streak"
    ]
    X = df[features].fillna(0)
    y_tx = df["tai"]
    y_cl = df["chan"]
    y_bao = df["bao"]

    models = {}
    for key, y in [("tx", y_tx), ("cl", y_cl), ("bao", y_bao)]:
        if len(set(y)) < 2:
            models[key] = None
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression().fit(X, y)
            rf = RandomForestClassifier(n_estimators=100).fit(X, y)
            xgbc = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss").fit(X, y)
        models[key] = (lr, rf, xgbc)

    joblib.dump(models, MODEL_PATH)
    with open(MODEL_META, "w") as f:
        f.write(str(len(df)))

def load_models():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def predict_stacking(X_pred, models, key):
    if models[key] is None:
        return 0.5, [0.5, 0.5, 0.5]
    lr, rf, xgbc = models[key]
    prob_lr = lr.predict_proba(X_pred)[0][1]
    prob_rf = rf.predict_proba(X_pred)[0][1]
    prob_xgb = xgbc.predict_proba(X_pred)[0][1]
    probs = np.array([prob_lr, prob_rf, prob_xgb])
    return probs.mean(), probs

# === MARKOV: CHUYỂN TRẠNG THÁI T/X + STREAK ÉP ĐẢO ===
def compute_markov_transition(df, window=MARKOV_WINDOW):
    df = df.tail(window)
    if len(df) < 10:
        return None
    seq = df["tai"].tolist()
    transitions = {"T->T": 0, "T->X": 0, "X->T": 0, "X->X": 0}
    for i in range(1, len(seq)):
        prev = "T" if seq[i - 1] == 1 else "X"
        curr = "T" if seq[i] == 1 else "X"
        transitions[f"{prev}->{curr}"] += 1

    total_T = transitions["T->T"] + transitions["T->X"]
    total_X = transitions["X->T"] + transitions["X->X"]
    prob_T2X = transitions["T->X"] / total_T if total_T else 0.0
    prob_X2T = transitions["X->T"] / total_X if total_X else 0.0
    last = "T" if seq[-1] == 1 else "X"

    return {
        "prob_T2X": prob_T2X,
        "prob_X2T": prob_X2T,
        "last": last
    }
# === PHẦN 4: TELEGRAM HANDLERS – XỬ LÝ NHẬP, DỰ ĐOÁN ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Chào bạn! Gửi kết quả 3 số (VD: 1 3 6) để BOT học và dự đoán!")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    await update.message.reply_text("✅ Đã reset phiên chơi!")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    session_start = load_session_start()
    if session_start:
        df_session = df[df['created_at'] >= session_start]
    else:
        df_session = df

    if len(df_session[df_session['input'] != "BOT_PREDICT"]) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"⚠️ Cần nhập tối thiểu {MIN_SESSION_INPUT} phiên để dự đoán!")
        return

    df_feat = make_features(df)
    df_feat_session = make_features(df_session)
    models = load_models()

    features = ['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
                'tai_lag_1', 'tai_lag_2', 'tai_lag_3', 'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
                'tai_streak', 'chan_streak']
    X_pred = df_feat_session.iloc[[-1]][features].fillna(0)

    tx_proba, _ = predict_stacking(X_pred, models, 'tx')
    cl_proba, _ = predict_stacking(X_pred, models, 'cl')
    tx = "Tài" if tx_proba >= 0.5 else "Xỉu"
    cl = "Chẵn" if cl_proba >= 0.5 else "Lẻ"

    markov_info = compute_markov_transition(df_feat_session)
    decision_override = False
    reason = ""
    tai_streak = df_feat_session.iloc[-1]['tai_streak']

    if markov_info:
        last = markov_info['last']
        p_T2X = markov_info['prob_T2X']
        p_X2T = markov_info['prob_X2T']
        if last == "T" and tx == "Tài" and p_T2X > p_X2T:
            tx = "Xỉu"
            decision_override = True
            reason = f"Markov: Tài→Xỉu ({p_T2X:.2f} > {p_X2T:.2f})"
        elif last == "X" and tx == "Xỉu" and p_X2T > p_T2X:
            tx = "Tài"
            decision_override = True
            reason = f"Markov: Xỉu→Tài ({p_X2T:.2f} > {p_T2X:.2f})"

    if not decision_override:
        if tx == "Tài" and tai_streak >= STREAK_THRESHOLD:
            tx = "Xỉu"
            decision_override = True
            reason = f"Chuỗi Tài liên tiếp ({tai_streak})"
        elif tx == "Xỉu" and tai_streak >= STREAK_THRESHOLD:
            tx = "Tài"
            decision_override = True
            reason = f"Chuỗi Xỉu liên tiếp ({tai_streak})"

    insert_result("BOT_PREDICT", None, tx)

    so_du_doan, dung, sai, tile = summary_stats(fetch_history(10000))
    lines = []
    if decision_override:
        lines.append(f"🔄 BOT đảo cửa: {tx} ({reason})")
    else:
        lines.append(f"🎯 Dự đoán phiên tiếp: {tx} | {cl}")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    await update.message.reply_text('\\n'.join(lines))

# === NHẬP KẾT QUẢ THỰC TẾ VÀ DỰ ĐOÁN TỰ ĐỘNG ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input_str = update.message.text.strip()
    if not re.match(r"^\\d+\\s+\\d+\\s+\\d+$", input_str):
        await update.message.reply_text("❌ Vui lòng nhập đúng định dạng: 3 số cách nhau bởi dấu cách (VD: 1 4 6)")
        return

    numbers = list(map(int, input_str.split()))
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
    insert_result(input_str, actual, None, input_time)

    await predict(update, context)
# === PHẦN 5: KHỞI CHẠY POLLING + FLASK KEEP-ALIVE ===

if __name__ == "__main__":
    create_table()

    # Chạy Flask nền để tránh Render sleep
    threading.Thread(target=start_flask, daemon=True).start()

    # Khởi tạo và chạy Telegram bot
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running (polling)...")
    app.run_polling()
