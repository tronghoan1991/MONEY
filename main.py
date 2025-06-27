import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
from datetime import datetime
import re
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from flask import Flask
import threading
import warnings
import traceback
import time
import io

# ==== CONFIG ====
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
INPUT_DELAY_SEC = 90   # nếu kết quả phiên nhập trễ hơn 90s, sẽ cảnh báo bỏ qua

if not BOT_TOKEN or not DATABASE_URL:
    print("Lỗi: Chưa set BOT_TOKEN hoặc DATABASE_URL.")
    raise Exception("Bạn cần set BOT_TOKEN và DATABASE_URL ở biến môi trường!")

# ==== FLASK giữ cổng để tránh sleep ====
def start_flask():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Bot is alive!", 200

    @app.route('/healthz')
    def health():
        return "OK", 200

    app.run(host='0.0.0.0', port=10000)

threading.Thread(target=start_flask, daemon=True).start()

# ==== DB ====
def create_table():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # ✅ Thêm cột input_time nếu chưa có
    try:
        cur.execute("ALTER TABLE history ADD COLUMN input_time FLOAT;")
        conn.commit()
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()

    # ✅ Tạo bảng nếu chưa có (giữ nguyên đoạn sau)
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


def insert_result(input_str, actual, bot_predict=None, input_time=None):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    now = datetime.now()
    if bot_predict is not None:
        cur.execute(
            "INSERT INTO history (input, actual, bot_predict, created_at, input_time) VALUES (%s, %s, %s, %s, %s);",
            (input_str, actual, bot_predict, now, input_time)
        )
    else:
        cur.execute(
            "INSERT INTO history (input, actual, created_at, input_time) VALUES (%s, %s, %s, %s);",
            (input_str, actual, now, input_time)
        )
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=10000):
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT id, input, actual, bot_predict, created_at, input_time FROM history ORDER BY id ASC LIMIT %s" % limit,
        engine
    )
    engine.dispose()
    df = df[df['input'].str.match(r"^\d+\s+\d+\s+\d+$", na=False) | (df['input'] == "BOT_PREDICT")]
    return df

# ==== Xuất toàn bộ lịch sử (KHÔNG giới hạn dòng) ====
def fetch_history_all():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT id, input, actual, bot_predict, created_at, input_time FROM history ORDER BY id ASC",
        engine
    )
    engine.dispose()
    df = df[df['input'].str.match(r"^\d+\s+\d+\s+\d+$", na=False) | (df['input'] == "BOT_PREDICT")]
    return df

def delete_all_history():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE history;")
    conn.commit()
    cur.close()
    conn.close()

# ==== FEATURE ENGINEERING ====
def make_features(df):
    df = df[df['input'].str.match(r"^\d+\s+\d+\s+\d+$", na=False)].copy()
    df['total'] = df['input'].apply(lambda x: sum([int(i) for i in x.split()]))
    df['even'] = df['total'] % 2
    df['bao'] = df['input'].apply(lambda x: 1 if len(set(x.split()))==1 else 0)
    df['tai'] = (df['total'] >= 11).astype(int)
    df['xiu'] = (df['total'] <= 10).astype(int)
    df['chan'] = (df['even'] == 0).astype(int)
    df['le'] = (df['even'] == 1).astype(int)

    roll_n = ROLLING_WINDOW
    df['tai_roll'] = df['tai'].rolling(roll_n, min_periods=1).mean()
    df['xiu_roll'] = df['xiu'].rolling(roll_n, min_periods=1).mean()
    df['chan_roll'] = df['chan'].rolling(roll_n, min_periods=1).mean()
    df['le_roll'] = df['le'].rolling(roll_n, min_periods=1).mean()
    df['bao_roll'] = df['bao'].rolling(roll_n, min_periods=1).mean()

    # Lag features
    for i in range(1, 4):
        df[f'tai_lag_{i}'] = df['tai'].shift(i)
        df[f'chan_lag_{i}'] = df['chan'].shift(i)

    # Streak features
    def get_streak(arr):
        streaks = [1]
        for i in range(1, len(arr)):
            if arr[i] == arr[i-1]:
                streaks.append(streaks[-1] + 1)
            else:
                streaks.append(1)
        return streaks
    df['tai_streak'] = get_streak(df['tai'].tolist())
    df['chan_streak'] = get_streak(df['chan'].tolist())
    return df

def train_models(df):
    df = df.tail(ROLLING_WINDOW*10)
    features = ['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
                'tai_lag_1', 'tai_lag_2', 'tai_lag_3', 'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
                'tai_streak', 'chan_streak']
    X = df[features].fillna(0)
    y_tx = (df['total'] >= 11).astype(int)
    y_cl = (df['even'] == 0).astype(int)
    y_bao = df['bao']
    models = {}
    for key, y in [('tx', y_tx), ('cl', y_cl), ('bao', y_bao)]:
        if len(set(y)) < 2:
            models[key] = None
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression().fit(X, y)
            rf = RandomForestClassifier(n_estimators=100).fit(X, y)
            xgbc = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss').fit(X, y)
        models[key] = (lr, rf, xgbc)
    joblib.dump(models, MODEL_PATH)
    with open(MODEL_META, "w") as f:
        f.write(str(len(df)))

def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def summary_stats(df):
    if 'bot_predict' in df.columns:
        df_pred = df[(df['bot_predict'].notnull()) & (df['actual'].notnull())]
        so_du_doan = len(df_pred)
        dung = (df_pred['bot_predict'] == df_pred['actual']).sum()
        sai = so_du_doan - dung
        tile = round((dung / so_du_doan) * 100, 2) if so_du_doan else 0
        return so_du_doan, dung, sai, tile
    else:
        return 0, 0, 0, 0

# ==== MARKOV CHAIN DỰ ĐOÁN ĐẢO CẦU (tự động học, không đặt ngưỡng cứng) ====
def compute_markov_transition(df):
    if len(df) < 10:
        return None
    seq = df['tai'].tolist()
    transitions = {"T->T":0, "T->X":0, "X->T":0, "X->X":0}
    for i in range(1, len(seq)):
        prev = "T" if seq[i-1] == 1 else "X"
        curr = "T" if seq[i] == 1 else "X"
        transitions[f"{prev}->{curr}"] += 1
    total_T = transitions["T->T"] + transitions["T->X"]
    total_X = transitions["X->T"] + transitions["X->X"]
    prob_T2X = transitions["T->X"] / total_T if total_T else 0.0
    prob_X2T = transitions["X->T"] / total_X if total_X else 0.0
    last = "T" if seq[-1]==1 else "X"
    return {
        "prob_T2X": prob_T2X,
        "prob_X2T": prob_X2T,
        "last": last
    }

def suggest_best_totals(df, prediction):
    if prediction not in ("Tài", "Xỉu") or df.empty:
        return "-"
    recent = df.tail(ROLLING_WINDOW)
    totals = [sum(int(x) for x in s.split()) for s in recent['input'] if s and s != "BOT_PREDICT"]
    if totals:
        mean = np.mean(totals)
        std = np.std(totals)
        safe_range = [t for t in totals if (mean-std)<=t<=(mean+std)]
    else:
        safe_range = totals
    if prediction == "Tài":
        eligible = [t for t in safe_range if t >= 11]
    else:
        eligible = [t for t in safe_range if t <= 10]
    count = pd.Series(eligible).value_counts()
    if count.empty:
        return "-"
    best = count.index[:3].tolist()
    if not best:
        return "-"
    return f"{min(best)}–{max(best)}"

def predict_stacking(X_pred, models, key):
    if models[key] is None:
        return 0.5, [0.5, 0.5, 0.5]
    lr, rf, xgbc = models[key]
    prob_lr = lr.predict_proba(X_pred)[0][1]
    prob_rf = rf.predict_proba(X_pred)[0][1]
    prob_xgb = xgbc.predict_proba(X_pred)[0][1]
    probs = np.array([prob_lr, prob_rf, prob_xgb])
    return probs.mean(), probs

# ==== HANDLER SAFE WRAPPER ====
def safe_handler(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await func(update, context)
        except Exception as e:
            err = f"LỖI: {e}\n{traceback.format_exc()}"
            try:
                await update.message.reply_text("🤖 BOT gặp lỗi kỹ thuật:\n" + str(e))
            except:
                pass
            print(err)
    return wrapper

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    create_table()
    now = datetime.now()
    input_time = time.time()
    last_play = load_last_play()
    delay_warning = False
    if last_play:
        delta = (now - last_play).total_seconds()
        if delta > INPUT_DELAY_SEC:
            delay_warning = True
    save_last_play(now)
    if user_id in PENDING_RESET and PENDING_RESET[user_id]:
        if text.upper() == "XÓA HẾT":
            delete_all_history()
            PENDING_RESET[user_id] = False
            await update.message.reply_text("✅ Đã xóa toàn bộ dữ liệu lịch sử.")
        else:
            PENDING_RESET[user_id] = False
            await update.message.reply_text("❌ Hủy thao tác xóa.")
        return

    m = re.match(r"^(\d{3})$", text)
    m2 = re.match(r"^(\d+)\s+(\d+)\s+(\d+)$", text)
    if not (m or m2):
        await update.message.reply_text("Vui lòng nhập kết quả theo định dạng: 456 hoặc 4 5 6.")
        return
    if m:
        numbers = [int(x) for x in list(m.group(1))]
    else:
        numbers = [int(m2.group(1)), int(m2.group(2)), int(m2.group(3))]
    if any(n < 1 or n > 6 for n in numbers):
        await update.message.reply_text("Kết quả không hợp lệ. Mỗi số phải từ 1–6!")
        return
    input_str = f"{numbers[0]} {numbers[1]} {numbers[2]}"
    total = sum(numbers)
    actual = "Tài" if total >= 11 else "Xỉu"
    if delay_warning:
        await update.message.reply_text("⚠️ Phiên này bạn nhập quá trễ (trên 90s), BOT sẽ không sử dụng dữ liệu này để đảm bảo độ chính xác dự đoán cho phiên tiếp theo.")
        return

    df = fetch_history(10000)
    last_predict = None
    if len(df) > 0 and df.iloc[-1]['bot_predict']:
        last_predict = df.iloc[-1]['bot_predict']

    if len(df) > 0 and df.iloc[-1]['input'] == "BOT_PREDICT" and (df.iloc[-1]['actual'] is None or pd.isnull(df.iloc[-1]['actual'])):
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        last_id = int(df.iloc[-1]['id'])
        cur.execute("UPDATE history SET actual=%s WHERE id=%s;", (actual, last_id))
        conn.commit()
        cur.close()
        conn.close()
    else:
        insert_result(input_str, actual, last_predict, input_time)

    df = fetch_history(10000)
    session_start = load_session_start()
    if session_start:
        df_session = df[df['created_at'] >= session_start]
    else:
        df_session = df
    if len(df_session[df_session['input'] != "BOT_PREDICT"]) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"Bạn cần nhập tối đa {MIN_SESSION_INPUT} phiên mới (sau khi bắt đầu session) để bot bắt đầu dự đoán trend session hiện tại!")
        return

    df_feat = make_features(df)
    n_trained = 0
    if os.path.exists(MODEL_META):
        with open(MODEL_META, "r") as f:
            try:
                n_trained = int(f.read())
            except:
                n_trained = 0
    if len(df) >= MIN_BATCH and (len(df) - n_trained >= TRAIN_EVERY):
        train_models(df_feat)
    models = load_models()
    if (models is None or models['tx'] is None or models['cl'] is None):
        lines = [
            f"✔️ Đã lưu kết quả: {''.join(str(n) for n in numbers)}",
            "⚠️ Chưa đủ dữ liệu đa dạng để dự đoán (cần đủ cả Tài/Xỉu & Chẵn/Lẻ). Nhập thêm tổng thấp/cao, chẵn/lẻ để bot hoạt động chính xác!"
        ]
        await update.message.reply_text('\n'.join(lines))
        return

    df_feat_session = make_features(df_session)
    features = ['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
                'tai_lag_1', 'tai_lag_2', 'tai_lag_3', 'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
                'tai_streak', 'chan_streak']
    X_pred = df_feat_session.iloc[[-1]][features].fillna(0)
    tx_proba, tx_probs = predict_stacking(X_pred, models, 'tx')
    tx = "Tài" if tx_proba >= 0.5 else "Xỉu"
    cl_proba, cl_probs = predict_stacking(X_pred, models, 'cl')
    cl = "Chẵn" if cl_proba >= 0.5 else "Lẻ"
    dai_diem = suggest_best_totals(df_session, tx)
    bao_pct = "-"
    if models.get('bao') is not None:
        bao_proba, bao_probs = predict_stacking(X_pred, models, 'bao')
        bao_pct = round(bao_proba*100,2)
    else:
        bao_proba = None

    # ==== MARKOV tự động học và quyết định đảo cầu (KHÔNG dùng ngưỡng cứng) ====
    markov_info = compute_markov_transition(df_feat_session)
    decision_override = False
    reason = ""
    if markov_info:
        if markov_info['last'] == "T" and tx == "Tài" and markov_info['prob_T2X'] > markov_info['prob_X2T']:
            tx = "Xỉu"
            decision_override = True
            reason = f"Markov phát hiện khả năng đảo cầu từ Tài sang Xỉu tăng bất thường."
        elif markov_info['last'] == "X" and tx == "Xỉu" and markov_info['prob_X2T'] > markov_info['prob_T2X']:
            tx = "Tài"
            decision_override = True
            reason = f"Markov phát hiện khả năng đảo cầu từ Xỉu sang Tài tăng bất thường."

    insert_result("BOT_PREDICT", None, tx)
    so_du_doan, dung, sai, tile = summary_stats(fetch_history(10000))
    lines = []
    lines.append(f"✔️ Đã lưu kết quả: {''.join(str(n) for n in numbers)}")
    if decision_override:
        lines.append(f"🔄 BOT tự động đảo cửa: {tx} ({reason})")
    else:
        lines.append(f"🎯 Dự đoán phiên tiếp: {tx} | {cl}")
    if abs(tx_proba - 0.5) < 0.1:
        lines.append("⚠️ BOT nhận diện thấy xác suất không rõ ràng, nên cân nhắc nghỉ phiên này!")
    lines.append(f"Dải điểm nên đánh: {dai_diem}")
    if bao_pct != "-":
        lines.append(f"Xác suất ra bão: {bao_pct}%")
        if bao_proba and bao_proba >= BAO_CUTOFF and models['bao'] is not None:
            lines.append(f"❗️CẢNH BÁO: Xác suất bão cao ({bao_pct}%) – cân nhắc vào bão!")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    await update.message.reply_text('\n'.join(lines))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 Chào mừng đến với Sicbo ML Bot Nâng Cấp!\n\n"
        "Các lệnh hỗ trợ:\n"
        "/start – Xem hướng dẫn và danh sách lệnh\n"
        "/predict – Dự đoán phiên tiếp theo\n"
        "/stats – Thống kê hiệu suất dự đoán\n"
        "/reset – Xóa toàn bộ lịch sử data (cần xác nhận)\n"
        "/exportdata – Xuất toàn bộ lịch sử dự đoán ra file Excel\n"
        "Nhập 3 số kết quả (vd: 456 hoặc 4 5 6) để lưu và cập nhật model.\n"
        "BOT sẽ tự động phát hiện trend, đảo cầu, và cảnh báo khi xác suất đảo chiều tăng bất thường!"
        f"\nNếu nghỉ quá {SESSION_BREAK_MINUTES} phút, bot sẽ tự động yêu cầu nhập tối đa {MIN_SESSION_INPUT} phiên đầu để bắt lại trend session!"
    )
    await update.message.reply_text(msg)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    session_start = load_session_start()
    if session_start:
        df_session = df[df['created_at'] >= session_start]
    else:
        df_session = df
    if len(df_session[df_session['input'] != "BOT_PREDICT"]) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"Bạn cần nhập tối đa {MIN_SESSION_INPUT} phiên mới (sau khi bắt đầu session) để bot bắt đầu dự đoán trend session hiện tại!")
        return
    df_feat = make_features(df)
    n_trained = 0
    if os.path.exists(MODEL_META):
        with open(MODEL_META, "r") as f:
            try:
                n_trained = int(f.read())
            except:
                n_trained = 0
    if len(df) - n_trained >= TRAIN_EVERY:
        train_models(df_feat)
    models = load_models()
    if (models is None or models['tx'] is None or models['cl'] is None):
        await update.message.reply_text("⚠️ Chưa đủ dữ liệu đa dạng để dự đoán (lịch sử cần đủ cả Tài/Xỉu và Chẵn/Lẻ). Nhập thêm các tổng thấp và cao, tổng chẵn/lẻ để bot hoạt động chính xác!")
        return
    df_feat_session = make_features(df_session)
    features = ['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll',
                'tai_lag_1', 'tai_lag_2', 'tai_lag_3', 'chan_lag_1', 'chan_lag_2', 'chan_lag_3',
                'tai_streak', 'chan_streak']
    X_pred = df_feat_session.iloc[[-1]][features].fillna(0)
    tx_proba, _ = predict_stacking(X_pred, models, 'tx')
    cl_proba, _ = predict_stacking(X_pred, models, 'cl')
    tx = "Tài" if tx_proba >= 0.5 else "Xỉu"
    cl = "Chẵn" if cl_proba >= 0.5 else "Lẻ"
    dai_diem = suggest_best_totals(df_session, tx)
    bao_pct = "-"
    if models.get('bao') is not None:
        bao_proba, _ = predict_stacking(X_pred, models, 'bao')
        bao_pct = round(bao_proba*100,2)
    else:
        bao_proba = None
    markov_info = compute_markov_transition(df_feat_session)
    decision_override = False
    reason = ""
    if markov_info:
        if markov_info['last'] == "T" and tx == "Tài" and markov_info['prob_T2X'] > markov_info['prob_X2T']:
            tx = "Xỉu"
            decision_override = True
            reason = f"Markov phát hiện khả năng đảo cầu từ Tài sang Xỉu tăng bất thường."
        elif markov_info['last'] == "X" and tx == "Xỉu" and markov_info['prob_X2T'] > markov_info['prob_T2X']:
            tx = "Tài"
            decision_override = True
            reason = f"Markov phát hiện khả năng đảo cầu từ Xỉu sang Tài tăng bất thường."
    insert_result("BOT_PREDICT", None, tx)
    so_du_doan, dung, sai, tile = summary_stats(fetch_history(10000))
    lines = []
    if decision_override:
        lines.append(f"🔄 BOT tự động đảo cửa: {tx} ({reason})")
    else:
        lines.append(f"🎯 Dự đoán phiên tiếp: {tx} | {cl}")
    lines.append(f"Dải điểm nên đánh: {dai_diem}")
    if bao_pct != "-":
        lines.append(f"Xác suất ra bão: {bao_pct}%")
        if bao_proba and bao_proba >= BAO_CUTOFF and models['bao'] is not None:
            lines.append(f"❗️CẢNH BÁO: Xác suất bão cao ({bao_pct}%) – cân nhắc vào bão!")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    await update.message.reply_text('\n'.join(lines))

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    so_du_doan, dung, sai, tile = summary_stats(df)
    msg = (
        f"BOT đã dự đoán: {so_du_doan} phiên\n"
        f"Đúng: {dung}\n"
        f"Sai: {sai}\n"
        f"Tỉ lệ đúng: {tile}%"
    )
    await update.message.reply_text(msg)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    PENDING_RESET[user_id] = True
    await update.message.reply_text(
        "⚠️ Bạn có chắc chắn muốn xóa toàn bộ lịch sử data? "
        "Nếu chắc chắn, reply: XÓA HẾT\n"
        "Nếu không, nhập bất kỳ ký tự nào khác để hủy."
    )

# ==== LỆNH XUẤT DỮ LIỆU LỊCH SỬ ĐẦY ĐỦ ====
async def export_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history_all()  # Không giới hạn số dòng
    if df.empty:
        await update.message.reply_text("Không có dữ liệu để xuất.")
        return
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.rename_axis("STT", inplace=True)
    if "created_at" in df.columns:
        df['created_at'] = df['created_at'].astype(str)
    if "input_time" in df.columns:
        df['input_time'] = df['input_time'].astype(str)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
        df.to_excel(writer, index=True, encoding='utf-8')
    output.seek(0)
    await update.message.reply_document(document=output, filename="lich_su_du_doan_Sicbo.xlsx",
                                       caption="File lịch sử dự đoán (không lỗi font, đủ mọi phiên, mở bằng Excel đều được).")

def main():
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", safe_handler(start)))
    app.add_handler(CommandHandler("predict", safe_handler(predict)))
    app.add_handler(CommandHandler("stats", safe_handler(stats)))
    app.add_handler(CommandHandler("reset", safe_handler(reset)))
    app.add_handler(CommandHandler("exportdata", safe_handler(export_data)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, safe_handler(handle_message)))
    app.run_polling()

# ==== STATE HANDLING ====
PENDING_RESET = {}

def save_session_start(time=None):
    with open(SESSION_FILE, "w") as f:
        t = time if time else datetime.now().isoformat()
        f.write(str(t))

def load_session_start():
    if not os.path.exists(SESSION_FILE):
        return None
    with open(SESSION_FILE, "r") as f:
        t = f.read().strip()
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None

def save_last_play(time=None):
    with open(LAST_PLAY_FILE, "w") as f:
        t = time if time else datetime.now().isoformat()
        f.write(str(t))

def load_last_play():
    if not os.path.exists(LAST_PLAY_FILE):
        return None
    with open(LAST_PLAY_FILE, "r") as f:
        t = f.read().strip()
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None

if __name__ == "__main__":
    main()
