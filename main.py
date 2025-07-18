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
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
MIN_SESSION_INPUT = 10
POINTS = list(range(3, 19))

def create_table():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                username TEXT,
                guess_type TEXT,
                guess_points TEXT,
                input_result TEXT,
                input_total INT,
                is_bao INT,
                is_correct INT,
                is_skip INT,
                win_streak INT,
                switch_cua INT,
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

def save_prediction(user_id, username, guess_type, guess_points, input_result, input_total, is_bao, is_correct, is_skip, win_streak, switch_cua, ml_pred_type, ml_pred_points):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (user_id, username, guess_type, guess_points, input_result, input_total, is_bao, is_correct, is_skip, win_streak, switch_cua, ml_pred_type, ml_pred_points)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (user_id, username, guess_type, guess_points, input_result, input_total, is_bao, is_correct, is_skip, win_streak, switch_cua, ml_pred_type, ml_pred_points))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error("DB Save Error: %s", e)

def fetch_history(limit=1000):
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM history ORDER BY id DESC LIMIT %s" % limit, engine)
        engine.dispose()
        df = df.sort_values('id')
        return df
    except Exception as e:
        logger.error("DB Fetch Error: %s", e)
        return pd.DataFrame()

def make_group_features(df):
    df = df.tail(20)
    df = df[df["is_skip"] == 0]
    if df.empty:
        return pd.DataFrame([{"tai_rate": 0, "xiu_rate": 0, "switch_cua_rate": 0, "win_streak": 0, "mean_total": 0, "last_cua": 0}])
    df["total"] = pd.to_numeric(df["input_total"], errors="coerce").fillna(0)
    df["tai"] = (df["total"] >= 11).astype(int)
    df["xiu"] = (df["total"] <= 10).astype(int)
    df["switch_cua"] = df["guess_type"] != df["guess_type"].shift(1)
    win_streak = (df["is_correct"] == 1).astype(int).groupby((df["is_correct"] != 1).cumsum()).cumcount().max() or 0
    last_cua = {"Tài": 0, "Xỉu": 1, "Bão": 2}.get(df["guess_type"].iloc[-1], 0)
    features = {
        "tai_rate": df["tai"].mean(),
        "xiu_rate": df["xiu"].mean(),
        "switch_cua_rate": df["switch_cua"].mean(),
        "win_streak": win_streak,
        "mean_total": df["total"].mean(),
        "last_cua": last_cua
    }
    return pd.DataFrame([features])

def group_predict(df):
    df = df[df["is_skip"] == 0]
    if len(df) < MIN_SESSION_INPUT:
        return None, None
    X = make_group_features(df)
    y_cua = df["guess_type"].replace({"Tài": 0, "Xỉu": 1, "Bão": 2}).shift(-1).dropna()
    if len(y_cua) < 5:
        return None, None
    X_cua = make_group_features(df.iloc[:-1])
    clf_cua = RandomForestClassifier(n_estimators=30)
    clf_cua.fit(X_cua, y_cua)
    y_pred_cua = clf_cua.predict(X)[0]
    cua_text = ["Tài", "Xỉu", "Bão"][int(y_pred_cua)]
    win_rows = df[df["is_correct"] == 1]
    if not win_rows.empty:
        dai_freq = {}
        for dai in win_rows["guess_points"]:
            if not dai: continue
            for i in dai.split(","):
                if i.isdigit():
                    dai_freq[int(i)] = dai_freq.get(int(i), 0) + 1
        top_dai = sorted(dai_freq.items(), key=lambda x: -x[1])[:3]
        dai_suggest = [str(d[0]) for d in top_dai]
        dai_suggest_str = ", ".join(dai_suggest) if dai_suggest else ""
    else:
        dai_suggest_str = ""
    return cua_text, dai_suggest_str

user_state = {}

async def start_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or f"User{user_id}"
    user_state[user_id] = {'step': 'choose_type', 'username': username}
    keyboard = [["Tài", "Xỉu", "Bão"], ["Bỏ qua phiên này"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Chọn cửa dự đoán hoặc bỏ qua:", reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or f"User{user_id}"
    text = update.message.text.strip()
    state = user_state.get(user_id, {'username': username})

    if state.get('step') == 'choose_type':
        if text == "Bỏ qua phiên này":
            save_prediction(user_id, username, None, None, None, None, None, None, 1, None, None, "-", "-")
            await update.message.reply_text("Bạn đã chọn bỏ qua phiên này. Khi muốn chơi tiếp, nhấn /batdau hoặc đợi phiên tiếp theo.")
            user_state[user_id] = {'step': None, 'username': username}
            return
        if text not in ["Tài", "Xỉu", "Bão"]:
            await update.message.reply_text("Vui lòng chọn 'Tài', 'Xỉu', 'Bão' hoặc 'Bỏ qua phiên này'.")
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
            return
        elif text == "Bỏ qua":
            user_state[user_id]['guess_points'] = set()
            user_state[user_id]['step'] = 'input_result'
            await update.message.reply_text(
                f"Bạn đã chọn: {user_state[user_id]['guess_type']} (không dải điểm)\nNhập kết quả thực tế (vd: 2 4 5 hoặc 245):"
            )
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
            return
        total = sum(nums)
        guess_type = user_state[user_id]['guess_type']
        guess_points = user_state[user_id]['guess_points']
        is_bao = int(nums[0] == nums[1] == nums[2])

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

        if guess_points and len(guess_points) > 0:
            is_dai = total in guess_points
            dai_text = f"[{', '.join(str(x) for x in sorted(guess_points))}]"
        else:
            is_dai = None
            dai_text = None

        if guess_type == "Bão":
            correct = is_cua
        elif guess_points and len(guess_points) > 0:
            correct = is_dai
        else:
            correct = is_cua

        # Sequence: win_streak, switch_cua (pattern đổi cửa)
        df_all = fetch_history(limit=1000)
        df = df_all[df_all["is_skip"] == 0]
        win_streak = 1
        switch_cua = 0
        if not df.empty:
            if df["is_correct"].iloc[-1] == 1:
                win_streak = (df["is_correct"] == 1).astype(int).groupby((df["is_correct"] != 1).cumsum()).cumcount().iloc[-1] + 1
            else:
                win_streak = 1
            switch_cua = int(guess_type != df["guess_type"].iloc[-1])

        kq_text = f"Kết quả: {nums[0]} {nums[1]} {nums[2]} (Tổng {total})\n"
        kq_text += f"Kết quả cửa: {cua_text} ({'ĐÚNG' if is_cua else 'SAI'})\n"
        if dai_text is not None:
            kq_text += f"Kết quả dải số: {dai_text} ({'ĐÚNG' if is_dai else 'SAI'})"

        try:
            cua_pred, dai_pred = group_predict(df_all)
            if cua_pred:
                ml_text = f"\n🤖 BOT dự đoán phiên tiếp (nhóm):\n• Cửa: {cua_pred}"
                if dai_pred:
                    ml_text += f"\n• Dải điểm gợi ý: {dai_pred}"
                else:
                    ml_text += f"\n• Dải điểm gợi ý: (không xác định)"
                await update.message.reply_text(ml_text)
        except Exception as e:
            logger.error(f"Group ML error: {e}")

        save_prediction(
            user_id=user_id,
            username=username,
            guess_type=guess_type,
            guess_points=",".join(str(p) for p in sorted(guess_points)) if guess_points else "",
            input_result=" ".join(str(x) for x in nums),
            input_total=total,
            is_bao=is_bao,
            is_correct=int(correct),
            is_skip=0,
            win_streak=win_streak,
            switch_cua=switch_cua,
            ml_pred_type=cua_pred if 'cua_pred' in locals() else "-",
            ml_pred_points=dai_pred if 'dai_pred' in locals() else "-"
        )

        await update.message.reply_text(kq_text)
        user_state[user_id] = {'step': 'choose_type', 'username': username}
        keyboard = [["Tài", "Xỉu", "Bão"], ["Bỏ qua phiên này"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Bạn muốn nhập dự đoán tiếp cho phiên mới? Chọn cửa dự đoán hoặc bỏ qua:", reply_markup=reply_markup)
        return

    await update.message.reply_text("Nhấn /batdau để bắt đầu dự đoán phiên mới.")

app = FastAPI()
telegram_app = Application.builder().token(BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("batdau", start_prediction))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Bot Sicbo nhóm tối ưu ML + hành vi!\n"
        "/batdau - Bắt đầu dự đoán mới\n"
        "/thongke - Lịch sử nhóm\n"
        "/reset - Xóa lịch sử cá nhân\n"
        "/help - Hướng dẫn"
    )
    await update.message.reply_text(text)
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("help", start))

async def thongke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(limit=20)
    if df.empty:
        text = "Chưa có lịch sử nào!"
    else:
        text = "Lịch sử 10 phiên gần nhất của nhóm:\n"
        for idx, row in df.tail(10).iterrows():
            user = row.get("username", "-")
            guess = row.get("guess_type", "-")
            dai = row.get("guess_points", "-")
            kq = row.get("input_result", "-")
            total = row.get("input_total", "-")
            res = "Đúng" if row.get("is_correct") else "Sai"
            skip = " (Bỏ qua)" if row.get("is_skip") else ""
            text += f"{user}{skip}, chọn: {guess}, dải: [{dai}], KQ: {kq} (Tổng {total}) - {res}\n"
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
telegram_app.add_handler(CommandHandler("reset", reset))

@app.on_event("startup")
async def on_startup():
    create_table()
    await telegram_app.initialize()
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/webhook/{BOT_TOKEN}"
    await telegram_app.bot.set_webhook(url=webhook_url)

@app.post(f"/webhook/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

@app.get("/")
async def root():
    return {"status": "ok", "message": "Bot is alive."}
@app.head("/")
async def root_head():
    return Response(status_code=200)
