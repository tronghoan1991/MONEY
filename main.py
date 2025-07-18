import os
import sys
import logging
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from fastapi import FastAPI, Request
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)
from datetime import datetime
import joblib
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
MODEL_PATH = "ml_stack_point.joblib"
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

# ==== ML + FEATURES ====
def make_features(df):
    df = df.copy()
    def extract_numbers(x):
        if isinstance(x, str) and re.match(r"^\d+\s+\d+\s+\d+$", x):
            return list(map(int, x.split()))
        elif isinstance(x, str) and len(x) == 3 and x.isdigit():
            return [int(x[0]), int(x[1]), int(x[2])]
        return [0, 0, 0]
    df[["n1", "n2", "n3"]] = df["input_result"].apply(lambda x: pd.Series(extract_numbers(x)))
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

def make_user_behavior_features(df):
    df = df.copy()
    df["guess_tai"] = (df["guess_type"] == "Tài").astype(int)
    df["guess_xiu"] = (df["guess_type"] == "Xỉu").astype(int)
    df["guess_bao"] = (df["guess_type"] == "Bão").astype(int)
    for pt in range(3, 19):
        df[f"guess_point_{pt}"] = df["guess_points"].apply(lambda x: str(pt) in str(x).split(","))
    return df[
        ["guess_tai", "guess_xiu", "guess_bao"] +
        [f"guess_point_{pt}" for pt in range(3, 19)]
    ].astype(int)

FEATURES = [
    'n1', 'n2', 'n3', 'total', 'even', 'bao',
    'tai', 'xiu', 'chan', 'le',
    'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
    'tai_lag_1', 'tai_lag_2', 'tai_lag_3',
    'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
    'tai_streak', 'chan_streak'
]

def train_point_model(df, save_path=MODEL_PATH):
    try:
        X = df[FEATURES].fillna(0)
        y = df['total'].astype(int)
        all_classes = np.arange(3, 19)
        present_classes = np.unique(y)
        missing_classes = [c for c in all_classes if c not in present_classes]
        if missing_classes:
            X_dummy = pd.DataFrame(0, index=np.arange(len(missing_classes)), columns=FEATURES)
            y_dummy = pd.Series(missing_classes)
            X = pd.concat([X, X_dummy], ignore_index=True)
            y = pd.concat([y, y_dummy], ignore_index=True)
        le = LabelEncoder()
        le.fit(all_classes)
        y_encoded = le.transform(y)
        models = [
            ('xgb', xgb.XGBClassifier(n_estimators=60, use_label_encoder=False, eval_metric='mlogloss')),
            ('rf', RandomForestClassifier(n_estimators=60)),
            ('lr', LogisticRegression(max_iter=300, multi_class='auto'))
        ]
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        ensemble.fit(X, y_encoded)
        joblib.dump({'ensemble': ensemble, 'label_encoder': le}, save_path)
        logger.info("Trained & saved main ML model (data rows: %d)", len(df))
    except Exception as e:
        logger.error("Train ML model Error: %s", e)

def train_behavior_model(df):
    try:
        X = make_user_behavior_features(df)
        y = df['input_total'].astype(int)
        all_classes = np.arange(3, 19)
        le = LabelEncoder()
        le.fit(all_classes)
        y_encoded = le.transform(y)
        models = [
            ('rf', RandomForestClassifier(n_estimators=40)),
            ('lr', LogisticRegression(max_iter=200, multi_class='auto'))
        ]
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        ensemble.fit(X, y_encoded)
        logger.info("Trained user-behavior ML model (data rows: %d)", len(df))
        return ensemble, le
    except Exception as e:
        logger.error("Train user-behavior ML Error: %s", e)
        return None, None

def load_point_model():
    try:
        if os.path.exists(MODEL_PATH):
            obj = joblib.load(MODEL_PATH)
            if isinstance(obj, dict) and 'ensemble' in obj and 'label_encoder' in obj:
                return obj['ensemble'], obj['label_encoder']
            else:
                return obj, None
        else:
            logger.warning("ML model not found.")
            return None, None
    except Exception as e:
        logger.error("Load ML model Error: %s", e)
        return None, None

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
                    f"Bạn đã chọn: {user_state[user_id]['guess_type']} các điểm {points}\nNhập kết quả thực tế (vd: 2 4 5):"
                )
            else:
                await update.message.reply_text(
                    f"Bạn đã chọn: {user_state[user_id]['guess_type']} (không dải điểm)\nNhập kết quả thực tế (vd: 2 4 5):"
                )
            logger.info(f"User {user_id} chọn dải điểm: {points}")
            return
        elif text == "Bỏ qua":
            user_state[user_id]['guess_points'] = set()
            user_state[user_id]['step'] = 'input_result'
            await update.message.reply_text(
                f"Bạn đã chọn: {user_state[user_id]['guess_type']} (không dải điểm)\nNhập kết quả thực tế (vd: 2 4 5):"
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
        nums = [int(x) for x in text.split() if x.isdigit()]
        if len(nums) != 3 or not all(1 <= x <= 6 for x in nums):
            await update.message.reply_text("Vui lòng nhập đúng 3 số từ 1 đến 6 (vd: 2 4 5).")
            logger.warning(f"User {user_id} nhập sai format kết quả: {text}")
            return
        total = sum(nums)
        guess_type = user_state[user_id]['guess_type']
        guess_points = user_state[user_id]['guess_points']
        is_bao = int(nums[0] == nums[1] == nums[2])
        correct = False
        if guess_type == "Bão":
            correct = is_bao
        elif guess_points:
            correct = total in guess_points
        else:
            if guess_type == "Tài":
                correct = 11 <= total <= 18
            elif guess_type == "Xỉu":
                correct = 3 <= total <= 10

        kq_text = f"Kết quả: {nums[0]} {nums[1]} {nums[2]} (Tổng {total})"
        logger.info(f"User {user_id} nhập kết quả: {nums}, dự đoán {guess_type}, {guess_points}, đúng: {correct}")

        # ==== ML 2 nguồn ====
        df = fetch_history(limit=1000)
        ml_pred_type, ml_pred_points = "-", "-"
        try:
            if len(df) >= MIN_SESSION_INPUT:
                df_feat = make_features(df)
                if not os.path.exists(MODEL_PATH) or len(df_feat) % 50 == 0:
                    train_point_model(df_feat)
                ensemble, le = joblib.load(MODEL_PATH).values()
                X_pred = make_features(df.tail(ROLLING_WINDOW)).iloc[[-1]][FEATURES].fillna(0)
                proba1 = None
                if ensemble is not None:
                    proba_all = ensemble.predict_proba(X_pred)[0]
                    classes = le.inverse_transform(np.arange(len(proba_all)))
                    prob_dict_1 = {int(cls): float(prob) for cls, prob in zip(classes, proba_all)}
                    for pt in POINTS:
                        if pt not in prob_dict_1:
                            prob_dict_1[pt] = 0.0
                    proba1 = prob_dict_1
                behavior_model, behavior_le = train_behavior_model(df)
                proba2 = {pt: 0.0 for pt in POINTS}
                if behavior_model is not None:
                    X_user = make_user_behavior_features(df.tail(1))
                    proba_all2 = behavior_model.predict_proba(X_user)[0]
                    classes2 = behavior_le.inverse_transform(np.arange(len(proba_all2)))
                    prob_dict_2 = {int(cls): float(prob) for cls, prob in zip(classes2, proba_all2)}
                    for pt in POINTS:
                        if pt not in prob_dict_2:
                            prob_dict_2[pt] = 0.0
                    proba2 = prob_dict_2
                final_prob = {pt: ALPHA * proba1.get(pt, 0) + (1 - ALPHA) * proba2.get(pt, 0) for pt in POINTS}
                prob_tai = sum([final_prob.get(pt, 0) for pt in range(11, 19)])
                prob_xiu = sum([final_prob.get(pt, 0) for pt in range(3, 11)])
                ml_pred_type = "Tài" if prob_tai > prob_xiu else "Xỉu"
                g_range = suggest_best_range_point(final_prob, 3, 18, length=3)
                ml_pred_points = f"{g_range[0]}-{g_range[1]}"
                await update.message.reply_text(
                    f"🤖 BOT (cá nhân hóa + dữ liệu thực tế) dự đoán phiên tiếp:\n• Cửa: {ml_pred_type}\n• Dải điểm: {ml_pred_points}"
                )
                logger.info(f"ML prediction (type: {ml_pred_type}, range: {ml_pred_points})")
            else:
                await update.message.reply_text("BOT cần tối thiểu 10 phiên để dự đoán ML.")
                logger.info(f"Chưa đủ phiên để ML predict ({len(df)} phiên).")
        except Exception as e:
            logger.error(f"ML prediction error: {e}")

        save_prediction(
            user_id=user_id,
            guess_type=guess_type,
            guess_points=",".join(str(p) for p in sorted(guess_points)) if guess_points else "",
            input_result=" ".join(str(x) for x in nums),
            input_total=total,
            is_bao=is_bao,
            is_correct=int(correct),
            ml_pred_type=ml_pred_type,
            ml_pred_points=ml_pred_points
        )

        if correct:
            await update.message.reply_text(f"✅ Đúng! {kq_text}")
        else:
            await update.message.reply_text(f"❌ Sai! {kq_text}")

        user_state[user_id] = {'step': 'choose_type'}
        keyboard = [["Tài", "Xỉu", "Bão"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Bạn muốn nhập dự đoán tiếp cho phiên mới? Chọn cửa dự đoán:", reply_markup=reply_markup)
        return

    await update.message.reply_text("Nhấn /batdau để bắt đầu dự đoán phiên mới.")

# ==== FASTAPI WEBHOOK ====
app = FastAPI()

telegram_app = Application.builder().token(BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("batdau", start_prediction))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

@app.on_event("startup")
async def on_startup():
    logger.info("App starting up, creating DB table & setting webhook.")
    create_table()
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
