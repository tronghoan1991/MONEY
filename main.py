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
ADMIN_USER_IDS = [1372450798]  # <-- Thay bằng user_id thật của bạn!

def to_python_type(x):
    # Chuyển đổi numpy.int64/float64 thành int/float thường, giữ nguyên None hoặc str
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

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
    # Chuyển đổi tất cả biến đầu vào sang kiểu python gốc để tránh lỗi numpy.int64 với psycopg2
    user_id = to_python_type(user_id)
    input_total = to_python_type(input_total)
    is_bao = to_python_type(is_bao)
    is_correct = to_python_type(is_correct)
    is_skip = to_python_type(is_skip)
    win_streak = to_python_type(win_streak)
    switch_cua = to_python_type(switch_cua)
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (user_id, username, guess_type, guess_points, input_result, input_total, is_bao, is_correct, is_skip, win_streak, switch_cua, ml_pred_type, ml_pred_points)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
    df = df.copy()
    df["total"] = pd.to_numeric(df["input_total"], errors="coerce").fillna(0)
    df["tai"] = (df["total"] >= 11).astype(int)
    df["xiu"] = (df["total"] <= 10).astype(int)
    df["switch_cua"] = df["guess_type"] != df["guess_type"].shift(1)
    win_streaks = (df["is_correct"] == 1).astype(int).groupby((df["is_correct"] != 1).cumsum()).cumcount()
    last_cua = df["guess_type"].replace({"Tài": 0, "Xỉu": 1, "Bão": 2})
    features = pd.DataFrame({
        "tai_rate": df["tai"].rolling(5, min_periods=1).mean(),
        "xiu_rate": df["xiu"].rolling(5, min_periods=1).mean(),
        "switch_cua_rate": df["switch_cua"].rolling(5, min_periods=1).mean(),
        "win_streak": win_streaks,
        "mean_total": df["total"].rolling(5, min_periods=1).mean(),
        "last_cua": last_cua
    })
    return features

def group_predict(df):
    df = df[df["is_skip"] == 0].reset_index(drop=True)
    if len(df) < MIN_SESSION_INPUT:
        logger.info(f"Chưa đủ {MIN_SESSION_INPUT} phiên hợp lệ để ML dự đoán (đã có {len(df)})")
        return None, None
    y_cua = df["guess_type"].replace({"Tài": 0, "Xỉu": 1, "Bão": 2}).shift(-1).dropna()
    X = make_group_features(df).iloc[:-1]
    if len(y_cua) != len(X) or len(X) < 2:
        logger.info("Chưa đủ dữ liệu sliding window để train ML group_predict.")
        return None, None
    clf_cua = RandomForestClassifier(n_estimators=30)
    clf_cua.fit(X, y_cua)
    X_pred = make_group_features(df).iloc[[-1]]
    y_pred_cua = clf_cua.predict(X_pred)[0]
    cua_text = ["Tài", "Xỉu", "Bão"][int(y_pred_cua)]
    win_rows = df[df["is_correct"] == 1]
    dai_freq = {}
    for dai in win_rows["guess_points"]:
        if not dai: continue
        for i in dai.split(","):
            if i.isdigit():
                dai_freq[int(i)] = dai_freq.get(int(i), 0) + 1
    top_dai = sorted(dai_freq.items(), key=lambda x: -x[1])[:3]
    dai_suggest = [str(d[0]) for d in top_dai]
    dai_suggest_str = ", ".join(dai_suggest) if dai_suggest else ""
    return cua_text, dai_suggest_str

# ------------------ Thêm hàm xác định kết quả thực tế ------------------
def get_result_type(row):
    if row["is_bao"] == 1:
        return "Bão"
    elif 11 <= row["input_total"] <= 18:
        return "Tài"
    elif 3 <= row["input_total"] <= 10:
        return "Xỉu"
    else:
        return None
# -----------------------------------------------------------------------

user_state = {}
pending_reset = {}
pending_resetdb = {}

async def start_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or f"User{user_id}"
    user_state[user_id] = {'step': 'choose_type', 'username': username}
    keyboard = [["Tài", "Xỉu", "Bão"], ["Bỏ qua phiên này"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Chọn cửa dự đoán hoặc bỏ qua:", reply_markup=reply_markup)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    pending_reset[user_id] = True
    await update.message.reply_text("Bạn chắc chắn muốn xóa lịch sử cá nhân? Trả lời 'Có' để xác nhận, hoặc gửi bất kỳ tin nhắn nào khác để hủy.")

async def resetdb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_USER_IDS:
        await update.message.reply_text("Bạn không có quyền thực hiện thao tác này.")
        return
    pending_resetdb[user_id] = True
    await update.message.reply_text("Bạn chắc chắn muốn xóa toàn bộ lịch sử? Trả lời 'Đồng ý' để xác nhận, hoặc gửi bất kỳ tin nhắn nào khác để hủy.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or f"User{user_id}"
    text = update.message.text.strip()
    state = user_state.get(user_id, {'username': username})

    # Xác nhận reset cá nhân
    if pending_reset.get(user_id, False):
        if text.lower() == "có":
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("DELETE FROM history WHERE user_id = %s", (user_id,))
            conn.commit()
            cur.close()
            conn.close()
            await update.message.reply_text("✅ Đã xóa lịch sử dự đoán của bạn.")
        else:
            await update.message.reply_text("Đã hủy thao tác xóa lịch sử.")
        pending_reset[user_id] = False
        return

    # Xác nhận reset database toàn bộ
    if pending_resetdb.get(user_id, False):
        if text.lower() == "đồng ý":
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("DELETE FROM history;")
            conn.commit()
            cur.close()
            conn.close()
            await update.message.reply_text("✅ Đã xóa toàn bộ lịch sử dự đoán!")
        else:
            await update.message.reply_text("Đã hủy thao tác xóa toàn bộ lịch sử.")
        pending_resetdb[user_id] = False
        return

    if state.get('step') == 'choose_type':
        if text == "Bỏ qua phiên này":
            save_prediction(
                user_id=user_id,
                username=username,
                guess_type=None,
                guess_points=None,
                input_result=None,
                input_total=None,
                is_bao=None,
                is_correct=None,
                is_skip=1,
                win_streak=0,
                switch_cua=0,
                ml_pred_type="-",
                ml_pred_points="-"
            )
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
        await update.message.reply_text("Chọn các điểm dự đoán (bấm số, tích đủ xong thì chọn 'Xác nhận'):", reply_markup=reply_markup)
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
            else:
                user_state[user_id]['guess_points'].add(point)
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
            else:
                valid_sessions = len(df_all[df_all["is_skip"] == 0])
                await update.message.reply_text(f"BOT cần nhập đủ {MIN_SESSION_INPUT} phiên KHÔNG bỏ qua để dự đoán ML (hiện tại có {valid_sessions}).")
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

        # ====== Thống kê đúng/sai của người chơi và ML nhóm (ĐÃ SỬA ĐÚNG) ======
        user_history = df_all[(df_all["user_id"] == user_id) & (df_all["is_skip"] == 0)]
        total_user = len(user_history)
        right_user = user_history["is_correct"].sum()
        user_right_pct = round(right_user / total_user * 100, 1) if total_user else 0

        df_ml = df_all[df_all["is_skip"] == 0].reset_index(drop=True)
        # ---- Thêm cột result_type (kết quả thực tế) ----
        if not df_ml.empty:
            df_ml["result_type"] = df_ml.apply(get_result_type, axis=1)
        ml_total = 0
        ml_right = 0

        if len(df_ml) > MIN_SESSION_INPUT:
            for i in range(MIN_SESSION_INPUT, len(df_ml)-1):
                X_train = make_group_features(df_ml.iloc[:i])
                y_train = df_ml["guess_type"].replace({"Tài": 0, "Xỉu": 1, "Bão": 2}).shift(-1).iloc[:i-1].dropna()
                if len(X_train) < 2 or len(y_train) != len(X_train.iloc[:-1]):
                    continue
                clf_cua = RandomForestClassifier(n_estimators=30)
                clf_cua.fit(X_train.iloc[:-1], y_train)
                X_pred = X_train.iloc[[-1]]
                # ---- SỬA: lấy y_true là kết quả thực tế, không phải dự đoán của người chơi ----
                y_true = df_ml["result_type"].replace({"Tài": 0, "Xỉu": 1, "Bão": 2}).iloc[i]
                y_pred = clf_cua.predict(X_pred)[0]
                if y_true in [0, 1, 2]:
                    ml_total += 1
                    if y_pred == y_true:
                        ml_right += 1

        ml_right_pct = round(ml_right / ml_total * 100, 1) if ml_total else 0

        stats_text = (
            f"\n📊 Thống kê dự đoán:\n"
            f"• Bạn: dự đoán {right_user}/{total_user} phiên đúng ({user_right_pct}%)\n"
            f"• ML nhóm: dự đoán {ml_right}/{ml_total} phiên đúng ({ml_right_pct}%)"
        )
        await update.message.reply_text(stats_text)
        # ====== Hết thống kê ======

        user_state[user_id] = {'step': 'choose_type', 'username': username}
        keyboard = [["Tài", "Xỉu", "Bão"], ["Bỏ qua phiên này"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Bạn muốn nhập dự đoán tiếp cho phiên mới? Chọn cửa dự đoán hoặc bỏ qua:", reply_markup=reply_markup)
        return

    await update.message.reply_text("Nhấn /batdau để bắt đầu dự đoán phiên mới.")

app = FastAPI()
telegram_app = Application.builder().token(BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("batdau", start_prediction))
telegram_app.add_handler(CommandHandler("reset", reset))
telegram_app.add_handler(CommandHandler("resetdb", resetdb))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Bot Sicbo nhóm tối ưu ML + hành vi!\n"
        "/batdau - Bắt đầu dự đoán mới\n"
        "/thongke - Lịch sử nhóm\n"
        "/reset - Xóa lịch sử cá nhân\n"
        "/resetdb - Xóa toàn bộ lịch sử (admin)\n"
        "/help - Hướng dẫn"
    )
    await update.message.reply_text(text)
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("help", start))

async def thongke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(limit=50)
    if df.empty:
        text = "Chưa có lịch sử nào!"
    else:
        df_valid = df[df["is_skip"] == 0]
        text = "Lịch sử 10 phiên KHÔNG BỎ QUA gần nhất của nhóm:\n"
        for idx, row in df_valid.tail(10).iterrows():
            user = row.get("username", "-")
            guess = row.get("guess_type", "-")
            dai = row.get("guess_points", "-")
            kq = row.get("input_result", "-")
            total = row.get("input_total", "-")
            res = "Đúng" if row.get("is_correct") else "Sai"
            text += f"{user}, chọn: {guess}, dải: [{dai}], KQ: {kq} (Tổng {total}) - {res}\n"
        text += f"\n(Tổng số phiên hợp lệ: {len(df_valid)})"
    await update.message.reply_text(text)
telegram_app.add_handler(CommandHandler("thongke", thongke))

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
