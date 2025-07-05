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
import numpy as np
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

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
LAST_N_ACCURACY = 50

def start_flask():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Bot is alive!", 200

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

def create_table():
    try:
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
    except Exception as e:
        print("Lỗi khi tạo database:", e)

def get_time_block():
    hour = datetime.now().hour
    if 5 <= hour <= 11:
        return 'sang'
    elif 12 <= hour <= 17:
        return 'chieu'
    else:
        return 'toi'

def insert_result_return_id(input_str, actual, bot_predict=None, input_time=None):
    result_id = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        now = datetime.now()
        if bot_predict is not None:
            cur.execute("""
                INSERT INTO history (input, actual, bot_predict, created_at, input_time)
                VALUES (%s, %s, %s, %s, %s) RETURNING id;
            """, (input_str, actual, bot_predict, now, input_time))
        else:
            cur.execute("""
                INSERT INTO history (input, actual, created_at, input_time)
                VALUES (%s, %s, %s, %s) RETURNING id;
            """, (input_str, actual, now, input_time))
        result_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Lỗi khi ghi lịch sử:", e)
    return result_id

def update_prev_bot_predict_actual(actual_value, id_current):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            UPDATE history
            SET actual = %s
            WHERE input = 'BOT_PREDICT' AND id < %s AND actual IS NULL
            ORDER BY id DESC
            LIMIT 1
        """, (actual_value, id_current))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Lỗi cập nhật actual cho BOT_PREDICT:", e)

def fetch_history(limit=10000):
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM history ORDER BY id ASC LIMIT %s" % limit, engine)
        engine.dispose()
        return df[df["input"].str.match(r"^\d+$|^\d+\s+\d+\s+\d+$", na=False) | (df["input"] == "BOT_PREDICT")]
    except Exception as e:
        print("Lỗi khi lấy lịch sử:", e)
        return pd.DataFrame()

def summary_stats(df):
    df_pred = df[df["input"] == "BOT_PREDICT"]
    correct = (df_pred["bot_predict"] == df_pred["actual"]).sum()
    total = len(df_pred)
    wrong = total - correct
    acc = round((correct / total) * 100, 2) if total else 0
    return total, correct, wrong, acc

def save_session_start(time=None):
    try:
        with open(SESSION_FILE, "w") as f:
            f.write(str(time if time else datetime.now().isoformat()))
    except Exception as e:
        print("Lỗi lưu session:", e)

def load_session_start():
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    except:
        return None

def save_last_play(time=None):
    try:
        with open(LAST_PLAY_FILE, "w") as f:
            f.write(str(time if time else datetime.now().isoformat()))
    except Exception as e:
        print("Lỗi lưu last_play:", e)

def load_last_play():
    if not os.path.exists(LAST_PLAY_FILE):
        return None
    try:
        with open(LAST_PLAY_FILE, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    except:
        return None

def make_features(df):
    df = df[df["input"].str.match(r"^\d+$|^\d+\s+\d+\s+\d+$", na=False)].copy()
    def extract_numbers(x):
        x = str(x)
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

FEATURES = [
    'total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
    'tai_lag_1', 'tai_lag_2', 'tai_lag_3',
    'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
    'tai_streak', 'chan_streak'
]

def train_models_by_timeblock(df, save_path=None):
    block = get_time_block()
    path = save_path if save_path else f"models_{block}.joblib"
    df = df.tail(ROLLING_WINDOW * 10)
    X = df[FEATURES].fillna(0)
    y_tx = df['tai']
    models = []
    models.append(LogisticRegression().fit(X, y_tx))
    models.append(RandomForestClassifier(n_estimators=100).fit(X, y_tx))
    models.append(xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss').fit(X, y_tx))
    joblib.dump(models, path)
    return path

def load_models_by_timeblock():
    block = get_time_block()
    path = f"models_{block}.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

def update_stacking_weights(df, models):
    if models is None or len(models) != 3:
        return [1/3, 1/3, 1/3]
    df = df.tail(LAST_N_ACCURACY)
    if len(df) < 10:
        return [1/3, 1/3, 1/3]
    X = df[FEATURES].fillna(0)
    y = df['tai']
    accs = []
    for model in models:
        try:
            preds = model.predict(X)
            acc = (preds == y).mean()
        except Exception:
            acc = 1/3
        accs.append(max(acc, 0.01))
    s = sum(accs)
    return [a/s for a in accs]

def compute_markov_transition(df):
    seq = df['tai'].dropna().astype(int).tolist()
    if len(seq) < 2:
        return None
    T2T = T2X = X2T = X2X = 0
    for i in range(1, len(seq)):
        prev, curr = seq[i-1], seq[i]
        if prev == 1 and curr == 1: T2T += 1
        elif prev == 1 and curr == 0: T2X += 1
        elif prev == 0 and curr == 1: X2T += 1
        elif prev == 0 and curr == 0: X2X += 1
    t_count = sum(1 for x in seq[:-1] if x == 1)
    x_count = sum(1 for x in seq[:-1] if x == 0)
    last = seq[-1]
    return {
        'last': 'T' if last == 1 else 'X',
        'prob_T2T': T2T / t_count if t_count else 0.5,
        'prob_T2X': T2X / t_count if t_count else 0.5,
        'prob_X2T': X2T / x_count if x_count else 0.5,
        'prob_X2X': X2X / x_count if x_count else 0.5
    }

def get_confidence_label(proba):
    if proba >= 0.8 or proba <= 0.2:
        return "Rất cao"
    elif proba >= 0.7 or proba <= 0.3:
        return "Cao"
    elif proba >= 0.6 or proba <= 0.4:
        return "Trung bình"
    else:
        return "Thấp"

def predict_stacking(X_pred, models, weights):
    probas = []
    for model in models:
        try:
            proba = model.predict_proba(X_pred)[:, 1][0]
        except Exception:
            proba = 0.5
        probas.append(proba)
    avg_proba = float(np.dot(probas, weights))
    return avg_proba, probas

def suggest_best_range(recent_totals, from_num, to_num, length=3):
    counts = {i: 0 for i in range(from_num, to_num+1)}
    for t in recent_totals:
        if from_num <= t <= to_num:
            counts[t] += 1
    best_range = (from_num, from_num+length-1)
    best_sum = sum([counts[i] for i in range(from_num, from_num+length)])
    for start in range(from_num, to_num-length+2):
        curr_sum = sum([counts[i] for i in range(start, start+length)])
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_range = (start, start+length-1)
    return best_range

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu_text = (
        "🎲 **BOT DỰ ĐOÁN SICBO - MENU LỆNH** 🎲\n"
        "\n"
        "Các lệnh & chức năng sẵn có:\n"
        "• Gửi 3 số phiên gần nhất (ví dụ: `354` hoặc `4 5 3`) để nhập kết quả thực tế và nhận dự đoán phiên tiếp theo\n"
        "• /start — Hiển thị menu các lệnh (bạn đang xem)\n"
        "• /reset — Reset lại phiên chơi, bắt đầu thống kê mới\n"
        "\n"
        "👉 **Hướng dẫn nhanh:**\n"
        "- Gửi đúng định dạng 3 số liên tiếp (vd: 345) hoặc cách nhau bởi dấu cách (vd: 3 4 5)\n"
        "- BOT sẽ lưu kết quả, tự động thống kê & dự đoán phiên kế tiếp.\n"
        "\n"
        "Chúc bạn may mắn và quản lý vốn thông minh! 🚦"
    )
    await update.message.reply_text(menu_text, parse_mode="Markdown")

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

    result_id = insert_result_return_id(" ".join(map(str, numbers)), actual, None, input_time)
    if result_id is not None:
        update_prev_bot_predict_actual(actual, result_id)
    await predict(update, context)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    session_start = load_session_start()
    df_session = df[df['created_at'] >= session_start] if session_start else df

    if len(df_session[df_session['input'] != "BOT_PREDICT"]) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"⚠️ Cần tối thiểu {MIN_SESSION_INPUT} phiên để dự đoán.")
        return

    df_feat = make_features(df)
    df_feat_session = make_features(df_session)

    if len(df_feat) >= 30 and (not os.path.exists(f"models_{get_time_block()}.joblib")):
        train_models_by_timeblock(df_feat)

    models = load_models_by_timeblock()
    if models is None or len(models) != 3:
        await update.message.reply_text("⚠️ Model chưa đủ dữ liệu huấn luyện, vui lòng nhập thêm phiên.")
        return

    weights = update_stacking_weights(df_feat, models)

    X_pred = df_feat_session.iloc[[-1]][FEATURES].fillna(0)
    tx_proba, probas = predict_stacking(X_pred, models, weights)
    decision = "Tài" if tx_proba >= 0.5 else "Xỉu"
    confidence = get_confidence_label(tx_proba)
    bao = int(X_pred["bao_roll"].values[0] > 0.2)
    streak = int(X_pred["tai_streak"].values[0])

    risk_note = ""
    if abs(tx_proba-0.5) < 0.1:
        risk_note = "⚠️ Xác suất quá thấp, không nên vào lệnh."
    elif abs(tx_proba-0.5) < 0.2:
        risk_note = "⚠️ Xác suất thấp, cân nhắc kỹ."

    override_reason = None
    markov_info = compute_markov_transition(df_feat_session)
    if markov_info and abs(tx_proba - 0.5) <= (1 - OVERRIDE_CUTOFF):
        last = markov_info['last']
        if last == "T" and decision == "Tài" and markov_info['prob_T2X'] > markov_info['prob_T2T']:
            decision = "Xỉu"
            override_reason = "Xác suất đảo chiều cao"
        elif last == "X" and decision == "Xỉu" and markov_info['prob_X2T'] > markov_info['prob_X2X']:
            decision = "Tài"
            override_reason = "Xác suất đảo chiều cao"

    if not override_reason and streak >= STREAK_THRESHOLD and abs(tx_proba - 0.5) <= (1 - OVERRIDE_CUTOFF):
        decision = "Xỉu" if decision == "Tài" else "Tài"
        override_reason = f"Chuỗi {('Tài' if decision == 'Xỉu' else 'Xỉu')} liên tiếp ({streak} phiên)"

    insert_result_return_id("BOT_PREDICT", None, decision)

    # --------- Gợi ý dải điểm nên đánh ----------
    recent_totals = df_feat_session['total'].dropna().astype(int).tolist()[-50:]
    if decision == "Xỉu":
        g_range = suggest_best_range(recent_totals, 4, 10, length=3)
        range_text = f"Nên đánh dải: {g_range[0]} – {g_range[1]}"
    else:
        g_range = suggest_best_range(recent_totals, 11, 17, length=3)
        range_text = f"Nên đánh dải: {g_range[0]} – {g_range[1]}"
    # -------------------------------------------

    total, correct, wrong, acc = summary_stats(df)
    result_msg = [f"BOT dự đoán phiên tiếp theo: {decision} (xác suất {tx_proba*100:.1f}%)"]
    if override_reason:
        result_msg.insert(1, f"Lý do: {override_reason}")
    result_msg.append(range_text)
    if risk_note:
        result_msg.append(risk_note)
    if bao:
        result_msg.append("⚡ Có thể vào bão")
    result_msg.append(f"Tổng phiên dự đoán: {total} | Đúng: {correct} | Sai: {wrong} | Chính xác: {acc}%")

    await update.message.reply_text("\n".join(result_msg))

if __name__ == "__main__":
    import asyncio
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    threading.Thread(target=start_flask, daemon=True).start()
    asyncio.run(app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False, stop_signals=None))
