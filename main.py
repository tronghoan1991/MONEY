import os
import sys
import logging
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from fastapi import FastAPI, Request, Response
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)
import joblib
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==== LOGGER ====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Starting bot with Python version: {sys.version}")

# ==== CONFIG ====
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
ROLLING_WINDOW = 12
MIN_SESSION_INPUT = 10
POINTS = list(range(3, 19))
ALPHA = 0.5

# ==== DB ====
def create_table():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                guess_type TEXT,
                guess_points TEXT,
                input_result TEXT,
                input_total INT,
                is_bao INT,
                is_correct INT,
                ml_pred_type TEXT,
                ml_pred_points TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("DB table 'history' checked/created.")
    except Exception as e:
        logger.error("DB Error (create_table): %s", e)

def save_prediction(user_id, guess_type, guess_points, input_result, input_total, is_bao, is_correct, ml_pred_type, ml_pred_points):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (user_id, guess_type, guess_points, input_result, input_total, is_bao, is_correct, ml_pred_type, ml_pred_points)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (user_id, guess_type, guess_points, input_result, input_total, is_bao, is_correct, ml_pred_type, ml_pred_points))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Saved prediction: user_id={user_id}, input={input_result}, guess={guess_type}, points={guess_points}")
    except Exception as e:
        logger.error("DB Save Error: %s", e)

def fetch_history(limit=1000):
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM history ORDER BY id DESC LIMIT %s" % limit, engine)
        engine.dispose()
        df = df.sort_values('id')
        logger.info(f"Fetched {len(df)} rows from history table.")
        return df
    except Exception as e:
        logger.error("DB Fetch Error: %s", e)
        return pd.DataFrame()

# ==== FEATURE ENGINEERING (CÁ NHÂN HÓA THỰC SỰ) ====
def make_personal_features(df_user):
    df = df_user.copy()
    df = df.tail(20)  # chỉ lấy 20 phiên gần nhất
    # 3 số kết quả thực tế
    def extract_numbers(x):
        found = re.findall(r"\d", str(x))
        arr = [int(i) for i in found if 1 <= int(i) <= 6]
        while len(arr) < 3:
            arr.append(0)
        return arr[:3]
    df[["n1", "n2", "n3"]] = df["input_result"].apply(lambda x: pd.Series(extract_numbers(x)))
    df["total"] = df[["n1", "n2", "n3"]].sum(axis=1)
    df["even"] = df["total"] % 2
    df["bao"] = df.apply(lambda row: 1 if row["n1"] == row["n2"] == row["n3"] else 0, axis=1)
    df["tai"] = (df["total"] >= 11).astype(int)
    df["xiu"] = (df["total"] <= 10).astype(int)
    df["chan"] = (df["even"] == 0).astype(int)
    df["le"] = (df["even"] == 1).astype(int)
    # rolling/streak
    for col in ["tai", "xiu", "chan", "le", "bao"]:
        df[f"{col}_roll"] = df[col].rolling(5, min_periods=1).mean()
    def get_streak(arr):
        streaks = [1]
        for i in range(1, len(arr)):
            if arr[i] == arr[i-1]:
                streaks.append(streaks[-1] + 1)
            else:
                streaks.append(1)
        return streaks
    df["tai_streak"] = get_streak(df["tai"].tolist())
    df["chan_streak"] = get_streak(df["chan"].tolist())
    # Hành vi chuyển cửa
    df["guess_Tai"] = (df["guess_type"] == "Tài").astype(int)
    df["guess_Xiu"] = (df["guess_type"] == "Xỉu").astype(int)
    df["guess_Bao"] = (df["guess_type"] == "Bão").astype(int)
    # Đổi cửa liên tiếp
    df["switch_cua"] = df["guess_type"] != df["guess_type"].shift(1)
    # Đúng/Sai
    df["win"] = df["is_correct"]
    # Chuỗi thắng/thua
    def win_streak(arr):
        streaks = [1]
        for i in range(1, len(arr)):
            if arr[i]:
                streaks.append(streaks[-1] + 1)
            else:
                streaks.append(1)
        return streaks
    df["win_streak"] = win_streak(df["win"].tolist())
    # Tần suất đổi dải
    df["guess_points_set"] = df["guess_points"].apply(lambda x: tuple(sorted([int(i) for i in str(x).split(",") if i.isdigit()])))
    df["switch_dai"] = df["guess_points_set"] != df["guess_points_set"].shift(1)
    # Thắng/thua liên tiếp
    win_count = (df["is_correct"] == 1).sum()
    lose_count = (df["is_correct"] == 0).sum()
    # Các đặc trưng tổng hợp:
    features = {
        "tai_rate": df["tai"].mean(),
        "xiu_rate": df["xiu"].mean(),
        "bao_rate": df["bao"].mean(),
        "chan_rate": df["chan"].mean(),
        "le_rate": df["le"].mean(),
        "switch_cua_rate": df["switch_cua"].mean(),
        "switch_dai_rate": df["switch_dai"].mean(),
        "win_rate": win_count / (win_count + lose_count + 1e-5),
        "mean_total": df["total"].mean(),
        "std_total": df["total"].std(),
        "last_tai": df["tai"].iloc[-1],
        "last_xiu": df["xiu"].iloc[-1],
        "last_bao": df["bao"].iloc[-1],
        "last_win": df["win"].iloc[-1],
        "win_streak": df["win_streak"].iloc[-1],
        "tai_streak": df["tai_streak"].iloc[-1],
        "chan_streak": df["chan_streak"].iloc[-1],
    }
    return pd.DataFrame([features])

# ==== ML: PERSONAL PREDICT ====
def personal_predict(df_user):
    # Nếu chưa đủ 10 phiên, không dự đoán
    if len(df_user) < MIN_SESSION_INPUT:
        return None, None
    X = make_personal_features(df_user)
    # Dự đoán cửa (Tài/Xỉu/Bão)
    # Dùng RF, dễ train nhanh
    y_cua = df_user["guess_type"].replace({"Tài": 0, "Xỉu": 1, "Bão": 2}).shift(-1).dropna()
    if len(y_cua) < 5:
        return None, None
    X_cua = make_personal_features(df_user.iloc[:-1])
    clf_cua = RandomForestClassifier(n_estimators=40)
    clf_cua.fit(X_cua, y_cua)
    y_pred_cua = clf_cua.predict(X)[0]
    cua_text = ["Tài", "Xỉu", "Bão"][int(y_pred_cua)]

    # Dự đoán dải điểm: Lấy dải điểm bạn thường chọn khi thắng, và gợi ý trung bình
    win_rows = df_user[df_user["is_correct"] == 1]
    if not win_rows.empty:
        dai_freq = {}
        for dai in win_rows["guess_points"]:
            if not dai: continue
            for i in dai.split(","):
                if i.isdigit():
                    dai_freq[int(i)] = dai_freq.get(int(i), 0) + 1
        top_dai = sorted(dai_freq.items(), key=lambda x: -x[1])[:3]
        dai_suggest = [str(d[0]) for d in top_dai]
        if dai_suggest:
            dai_suggest_str = ", ".join(dai_suggest)
        else:
            dai_suggest_str = ""
    else:
        dai_suggest_str = ""
    return cua_text, dai_suggest_str

# ==== BOT FLOW ====
user_state = {}

async def start_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state[user_id] = {'step': 'choose_type'}
    keyboard = [["Tài", "Xỉu", "Bão"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Chọn cửa dự đoán:", reply_markup=reply_markup)
    logger.info(f"User {user_id} bắt đầu dự đoán mới.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    state = user_state.get(user_id, {})

    if state.get('step') == 'choose_type':
        if text not in ["Tài", "Xỉu", "Bão"]:
            await update.message.reply_text("Vui lòng chọn 'Tài', 'Xỉu' hoặc 'Bão'.")
            return
        user_state[user_id]['guess_type'] = text
        user_state[user_id]['guess_points'] = set()
        user_state[user_id]['step'] = 'choose_points'
        keyboard = [
            [str(i) for i in range(3, 7)],
            [str(i) for i in range(7, 11)],
            [str(i) for i in range(11, 15)],
            [str(i) for i in range(15, 19)],
            ["Xác nhận", "Bỏ qua"]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)
        await update.message.reply_text("Chọn từng điểm dự đoán (bấm từng số, xong thì chọn 'Xác nhận'):", reply_markup=reply_markup)
        logger.info(f"User {user_id} chọn cửa: {text}")
        return

    if state.get('step') == 'choose_points':
        if text == "Xác nhận":
            user_state[user_id]['step'] = 'input_result'
            points = sorted(user_state[user_id]['guess_points'])
            if points:
                await update.message.reply_text(
                    f"Bạn đã chọn: {user_state[user_id]['guess_type']} các điểm {points}\nNhập kết quả thực tế (vd: 2 4 5 hoặc 245):"
                )
            else:
                await update.message.reply_text(
                    f"Bạn đã chọn: {user_state[user_id]['guess_type']} (không dải điểm)\nNhập kết quả thực tế (vd: 2 4 5 hoặc 245):"
                )
            logger.info(f"User {user_id} chọn dải điểm: {points}")
            return
        elif text == "Bỏ qua":
            user_state[user_id]['guess_points'] = set()
            user_state[user_id]['step'] = 'input_result'
            await update.message.reply_text(
                f"Bạn đã chọn: {user_state[user_id]['guess_type']} (không dải điểm)\nNhập kết quả thực tế (vd: 2 4 5 hoặc 245):"
            )
            logger.info(f"User {user_id} bỏ qua dải điểm.")
            return
        elif text.isdigit() and 3 <= int(text) <= 18:
            point = int(text)
            if point in user_state[user_id]['guess_points']:
                user_state[user_id]['guess_points'].remove(point)
                await update.message.reply_text(f"Bỏ chọn điểm {point}. Các điểm đã chọn: {sorted(user_state[user_id]['guess_points'])}")
            else:
                user_state[user_id]['guess_points'].add(point)
                await update.message.reply_text(f"Đã chọn thêm điểm {point}. Các điểm đã chọn: {sorted(user_state[user_id]['guess_points'])}")
            return
        else:
            await update.message.reply_text("Hãy bấm số điểm muốn chọn hoặc 'Xác nhận'.")
            return

    if state.get('step') == 'input_result':
        cleaned = re.findall(r'\d', text)
        nums = [int(x) for x in cleaned if 1 <= int(x) <= 6]
        if len(nums) != 3:
            await update.message.reply_text("Vui lòng nhập đúng 3 số từ 1 đến 6 (vd: 2 4 5 hoặc 245).")
            logger.warning(f"User {user_id} nhập sai format kết quả: {text}")
            return
        total = sum(nums)
        guess_type = user_state[user_id]['guess_type']
        guess_points = user_state[user_id]['guess_points']
        is_bao = int(nums[0] == nums[1] == nums[2])

        # Xác định kết quả cửa
        if guess_type == "Tài":
            is_cua = 11 <= total <= 18
            cua_text = "Tài"
        elif guess_type == "Xỉu":
            is_cua = 3 <= total <= 10
            cua_text = "Xỉu"
        elif guess_type == "Bão":
            is_cua = is_bao == 1
            cua_text = "Bão"
        else:
            is_cua = False
            cua_text = guess_type

        # Xác định kết quả dải số (nếu có chọn)
        if guess_points and len(guess_points) > 0:
            is_dai = total in guess_points
            dai_text = f"[{', '.join(str(x) for x in sorted(guess_points))}]"
        else:
            is_dai = None
            dai_text = None

        # ĐÚNG/SAI tổng thể: chỉ tính đúng nếu đúng cả 2
        if guess_type == "Bão":
            correct = is_cua
        elif guess_points and len(guess_points) > 0:
            correct = is_dai
        else:
            correct = is_cua

        kq_text = f"Kết quả: {nums[0]} {nums[1]} {nums[2]} (Tổng {total})\n"
        kq_text += f"Kết quả cửa: {cua_text} ({'ĐÚNG' if is_cua else 'SAI'})\n"
        if dai_text is not None:
            kq_text += f"Kết quả dải số: {dai_text} ({'ĐÚNG' if is_dai else 'SAI'})"

        # ML CÁ NHÂN HÓA
        df_all = fetch_history(limit=1000)
        df_user = df_all[df_all['user_id'] == user_id]
        try:
            if len(df_user) >= MIN_SESSION_INPUT:
                cua_pred, dai_pred = personal_predict(df_user)
                if cua_pred:
                    ml_text = f"\n🤖 BOT dự đoán phiên tiếp:\n• Cửa: {cua_pred}"
                    if dai_pred:
                        ml_text += f"\n• Dải điểm gợi ý: {dai_pred}"
                    else:
                        ml_text += f"\n• Dải điểm gợi ý: (không xác định)"
                    await update.message.reply_text(ml_text)
        except Exception as e:
            logger.error(f"Personal ML error: {e}")

        save_prediction(
            user_id=user_id,
            guess_type=guess_type,
            guess_points=",".join(str(p) for p in sorted(guess_points)) if guess_points else "",
            input_result=" ".join(str(x) for x in nums),
            input_total=total,
            is_bao=is_bao,
            is_correct=int(correct),
            ml_pred_type=cua_pred if 'cua_pred' in locals() else "-",
            ml_pred_points=dai_pred if 'dai_pred' in locals() else "-"
        )

        await update.message.reply_text(kq_text)
        user_state[user_id] = {'step': 'choose_type'}
        keyboard = [["Tài", "Xỉu", "Bão"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Bạn muốn nhập dự đoán tiếp cho phiên mới? Chọn cửa dự đoán:", reply_markup=reply_markup)
        return

    await update.message.reply_text("Nhấn /batdau để bắt đầu dự đoán phiên mới.")

# ==== FASTAPI WEBHOOK & BOT COMMANDS ====
app = FastAPI()

telegram_app = Application.builder().token(BOT_TOKEN).build()

telegram_app.add_handler(CommandHandler("batdau", start_prediction))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# -- Thêm lệnh /start, /help, /thongke, /reset --
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Chào mừng bạn đến với bot dự đoán cá nhân hóa!\n"
        "Các lệnh hỗ trợ:\n"
        "/batdau - Bắt đầu dự đoán mới\n"
        "/thongke - Xem thống kê lịch sử dự đoán\n"
        "/reset - Xóa toàn bộ lịch sử dự đoán của bạn\n"
        "/help - Xem hướng dẫn sử dụng\n"
    )
    await update.message.reply_text(text)

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("help", start))

async def thongke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    df = fetch_history(limit=1000)
    df_user = df[df['user_id'] == user_id]
    total = len(df_user)
    correct = df_user['is_correct'].sum()
    text = (
        f"📊 Thống kê cá nhân:\n"
        f"- Số phiên nhập: {total}\n"
        f"- Số lần đúng: {correct}\n"
        f"- Tỉ lệ đúng: {round(100 * correct / total, 2) if total else 0}%"
    )
    await update.message.reply_text(text)

telegram_app.add_handler(CommandHandler("thongke", thongke))

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("DELETE FROM history WHERE user_id = %s", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
        await update.message.reply_text("✅ Đã xóa lịch sử dự đoán của bạn.")
    except Exception as e:
        await update.message.reply_text("Có lỗi khi xóa dữ liệu. Vui lòng thử lại.")
        logger.error("DB Reset Error: %s", e)

telegram_app.add_handler(CommandHandler("reset", reset))

# -- FIX lỗi initialize --
@app.on_event("startup")
async def on_startup():
    logger.info("App starting up, creating DB table & setting webhook.")
    create_table()
    await telegram_app.initialize()
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/webhook/{BOT_TOKEN}"
    await telegram_app.bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set: {webhook_url}")

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    logger.info(f"Received update: {data}")
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

# -- Chống sleep (UptimeRobot ping) + fix HEAD 405 --
@app.get("/")
async def root():
    return {"status": "ok", "message": "Bot is alive."}

@app.head("/")
async def root_head():
    return Response(status_code=200)
