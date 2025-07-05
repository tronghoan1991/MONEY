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
MODEL_PATH = "ml_stack_point.joblib"
SESSION_FILE = "session_time.txt"
LAST_PLAY_FILE = "last_play_time.txt"

MIN_SESSION_INPUT = 10
SESSION_BREAK_MINUTES = 30

POINTS = list(range(4, 18))  # tổng điểm từ 4 tới 17
TRAIN_LIMIT = 2000           # rolling window phù hợp RAM free server

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

def fetch_history(limit=TRAIN_LIMIT):
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM history ORDER BY id DESC LIMIT %s" % limit, engine)
        engine.dispose()
        df = df.sort_values('id')  # đảm bảo đúng thứ tự thời gian
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
    'n1', 'n2', 'n3', 'total', 'even', 'bao',
    'tai', 'xiu', 'chan', 'le',
    'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
    'tai_lag_1', 'tai_lag_2', 'tai_lag_3',
    'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
    'tai_streak', 'chan_streak'
]

def train_point_model(df, save_path=MODEL_PATH):
    X = df[FEATURES].fillna(0)
    y = df['total'].astype(int)
    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    joblib.dump(model, save_path)
    return None

def load_point_model():
    path = MODEL_PATH
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

def get_confidence_label(proba):
    if proba >= 0.8 or proba <= 0.2:
        return "Rất cao"
    elif proba >= 0.7 or proba <= 0.3:
        return "Cao"
    elif proba >= 0.6 or proba <= 0.4:
        return "Trung bình"
    else:
        return "Thấp"

def suggest_best_range_point(pred_prob_dict, from_num, to_num, length=3):
    best_range = (from_num, from_num+length-1)
    keys = list(range(from_num, to_num+1))
    vals = [pred_prob_dict.get(k, 0) for k in keys]
    best_sum = sum(vals[:length])
    for i in range(0, len(vals)-length+1):
        curr_sum = sum(vals[i:i+length])
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_range = (keys[i], keys[i+length-1])
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
    df = fetch_history()
    session_start = load_session_start()
    df_session = df[df['created_at'] >= session_start] if session_start else df

    if len(df_session[df_session['input'] != "BOT_PREDICT"]) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"⚠️ Cần tối thiểu {MIN_SESSION_INPUT} phiên để dự đoán.")
        return

    df_feat = make_features(df)
    df_feat_session = make_features(df_session)

    model = load_point_model()
    if model is None or (len(df_feat) % 100 == 0 and len(df_feat) > 0):
        train_point_model(df_feat)
        model = load_point_model()

    if model is None:
        await update.message.reply_text("⚠️ Model chưa đủ dữ liệu huấn luyện, vui lòng nhập thêm phiên.")
        return

    X_pred = df_feat_session.iloc[[-1]][FEATURES].fillna(0)
    try:
        proba_all = model.predict_proba(X_pred)[0]
        classes = model.classes_
    except Exception:
        await update.message.reply_text("⚠️ Model lỗi, cần nhập thêm phiên hoặc retrain.")
        return
    prob_dict = {int(cls): float(prob) for cls, prob in zip(classes, proba_all)}
    # Nếu thiếu tổng điểm nào thì cho xác suất = 0 (không bao giờ báo lỗi)
    for pt in range(4, 18):
        if pt not in prob_dict:
            prob_dict[pt] = 0.0

    prob_tai = sum([prob_dict.get(pt, 0) for pt in range(11, 18)])
    prob_xiu = sum([prob_dict.get(pt, 0) for pt in range(4, 11)])

    if prob_tai >= prob_xiu:
        decision = "Tài"
        tx_proba = prob_tai
        g_range = suggest_best_range_point(prob_dict, 11, 17, length=3)
        range_text = f"Nên đánh dải: {g_range[0]} – {g_range[1]}"
    else:
        decision = "Xỉu"
        tx_proba = prob_xiu
        g_range = suggest_best_range_point(prob_dict, 4, 10, length=3)
        range_text = f"Nên đánh dải: {g_range[0]} – {g_range[1]}"

    confidence = get_confidence_label(tx_proba)
    risk_note = ""
    if abs(tx_proba-0.5) < 0.1:
        risk_note = "⚠️ Xác suất quá thấp, không nên vào lệnh."
    elif abs(tx_proba-0.5) < 0.2:
        risk_note = "⚠️ Xác suất thấp, cân nhắc kỹ."

    bao = int(X_pred["bao_roll"].values[0] > 0.2)
    streak = int(X_pred["tai_streak"].values[0])

    insert_result_return_id("BOT_PREDICT", None, decision)

    total, correct, wrong, acc = summary_stats(df)
    result_msg = [f"BOT dự đoán phiên tiếp theo: {decision} (xác suất {tx_proba*100:.1f}%)"]
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
