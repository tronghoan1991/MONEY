import os
import pandas as pd
import psycopg2
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime
import re
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from flask import Flask
import threading

# ==== CONFIG ====
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
MIN_BATCH = 5
ROLLING_WINDOW = 50
PROBA_CUTOFF = 0.62
PROBA_ALERT = 0.75
BAO_CUTOFF = 0.03

MODEL_PATH = "ml_stack.joblib"

if not BOT_TOKEN or not DATABASE_URL:
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            input TEXT,
            actual TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_result(input_str, actual):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    now = datetime.now()
    cur.execute(
        "INSERT INTO history (input, actual, created_at) VALUES (%s, %s, %s);",
        (input_str, actual, now)
    )
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=10000):
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql("SELECT input, actual, created_at FROM history ORDER BY id ASC LIMIT %s" % limit, conn)
    conn.close()
    return df

def make_features(df):
    df = df.copy()
    df['total'] = df['input'].apply(lambda x: sum([int(i) for i in x.split()]))
    df['even'] = df['total'] % 2
    df['bao'] = df['input'].apply(lambda x: 1 if len(set(x.split()))==1 else 0)
    df['tai'] = (df['total'] >= 11).astype(int)
    df['xiu'] = (df['total'] <= 10).astype(int)
    df['chan'] = (df['even'] == 0).astype(int)
    df['le'] = (df['even'] == 1).astype(int)
    df['tai_roll'] = df['tai'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['xiu_roll'] = df['xiu'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['chan_roll'] = df['chan'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['le_roll'] = df['le'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['bao_roll'] = df['bao'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return df

def train_models(df):
    X = df[['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll']]
    y_tx = (df['total'] >= 11).astype(int)
    y_cl = (df['even'] == 0).astype(int)
    y_bao = df['bao']
    models = {}
    for key, y in [('tx', y_tx), ('cl', y_cl), ('bao', y_bao)]:
        if len(set(y)) < 2:
            # Không đủ 2 class, bỏ qua train, trả None
            models[key] = None
            continue
        lr = LogisticRegression().fit(X, y)
        rf = RandomForestClassifier(n_estimators=100).fit(X, y)
        xgbc = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss').fit(X, y)
        models[key] = (lr, rf, xgbc)
    joblib.dump(models, MODEL_PATH)

def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def predict_stacking(X_pred, models, key):
    if models[key] is None:
        # Trả về xác suất default và cảnh báo
        return 0.5, [0.5, 0.5, 0.5]
    lr, rf, xgbc = models[key]
    prob_lr = lr.predict_proba(X_pred)[0][1]
    prob_rf = rf.predict_proba(X_pred)[0][1]
    prob_xgb = xgbc.predict_proba(X_pred)[0][1]
    probs = np.array([prob_lr, prob_rf, prob_xgb])
    return probs.mean(), probs

def summary_stats(df):
    num = len(df)
    if num == 0:
        return 0, 0, 0, 0
    so_du_doan = num
    dung = 0
    sai = 0
    tile = 0
    return so_du_doan, dung, sai, tile

def suggest_best_totals(df, prediction):
    if prediction not in ("Tài", "Xỉu") or df.empty:
        return "-"
    recent = df.tail(ROLLING_WINDOW)
    totals = [sum(int(x) for x in s.split()) for s in recent['input'] if s]
    if prediction == "Tài":
        eligible = [t for t in range(11, 19)]
    else:
        eligible = [t for t in range(3, 11)]
    count = pd.Series([t for t in totals if t in eligible]).value_counts()
    if count.empty:
        return "-"
    best = count.index[:3].tolist()
    if not best:
        return "-"
    return f"{min(best)}–{max(best)}"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    create_table()
    m = re.match(r"^(\d{3})$", text)
    m2 = re.match(r"^(\d+)\s+(\d+)\s+(\d+)$", text)
    if not (m or m2):
        await update.message.reply_text("Vui lòng nhập kết quả theo định dạng: 456 hoặc 4 5 6.")
        return
    numbers = [int(x) for x in (m.group(1) if m else " ".join([m2.group(1), m2.group(2), m2.group(3)]))]
    input_str = f"{numbers[0]} {numbers[1]} {numbers[2]}"
    total = sum(numbers)
    actual = "Tài" if total >= 11 else "Xỉu"
    insert_result(input_str, actual)
    df = fetch_history(10000)
    df_feat = make_features(df)
    if len(df) >= MIN_BATCH:
        train_models(df_feat)
    models = load_models()
    if (models is None or 
        models['tx'] is None or
        models['cl'] is None or
        models['bao'] is None):
        lines = []
        lines.append(f"✔️ Đã lưu kết quả: {''.join(str(n) for n in numbers)}")
        lines.append("⚠️ Chưa đủ dữ liệu đa dạng để dự đoán (lịch sử mới chỉ có 1 loại kết quả). Nhập thêm cả Tài/Xỉu, Chẵn/Lẻ, Bão/Không bão để bot hoạt động chính xác!")
        await update.message.reply_text('\n'.join(lines))
        return
    X_pred = df_feat.iloc[[-1]][['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll']]
    tx_proba, tx_probs = predict_stacking(X_pred, models, 'tx')
    tx = "Tài" if tx_proba >= 0.5 else "Xỉu"
    cl_proba, cl_probs = predict_stacking(X_pred, models, 'cl')
    cl = "Chẵn" if cl_proba >= 0.5 else "Lẻ"
    dai_diem = suggest_best_totals(df, tx)
    bao_proba, bao_probs = predict_stacking(X_pred, models, 'bao')
    bao_pct = round(bao_proba*100,2)
    so_du_doan, dung, sai, tile = summary_stats(df)
    lines = []
    lines.append(f"✔️ Đã lưu kết quả: {''.join(str(n) for n in numbers)}")
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"🎯 Dự đoán: {tx} | {cl}")
    else:
        lines.append("⚠️ Dự đoán: Nên nghỉ phiên này!")
    lines.append(f"Dải điểm nên đánh: {dai_diem}")
    lines.append(f"Xác suất ra bão: {bao_pct}%")
    if max(tx_proba, 1-tx_proba) >= PROBA_ALERT:
        lines.append(f"❗️CẢNH BÁO: Xác suất {tx} vượt {int(PROBA_ALERT*100)}% – trend cực mạnh!")
    if bao_proba >= BAO_CUTOFF:
        lines.append(f"❗️CẢNH BÁO: Xác suất bão cao ({bao_pct}%) – cân nhắc vào bão!")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"Nhận định: Ưu tiên {tx}, {cl}, dải {dai_diem}. Bão {bao_pct}% – {'ưu tiên' if bao_proba >= BAO_CUTOFF else 'không nên đánh'} bão.")
    else:
        lines.append("Nhận định: Không có cửa ưu thế, nên nghỉ.")
    await update.message.reply_text('\n'.join(lines))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Đây là Sicbo ML Bot.\n"
        "- Nhập 3 số kết quả (vd: 456 hoặc 4 5 6) để lưu và cập nhật model.\n"
        "- Gõ /predict để nhận dự đoán phiên tiếp theo."
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    if len(df) < MIN_BATCH:
        await update.message.reply_text("Chưa đủ dữ liệu để dự đoán. Hãy nhập thêm kết quả!")
        return
    df_feat = make_features(df)
    train_models(df_feat)
    models = load_models()
    if (models is None or 
        models['tx'] is None or
        models['cl'] is None or
        models['bao'] is None):
        await update.message.reply_text("⚠️ Chưa đủ dữ liệu đa dạng để dự đoán (lịch sử mới chỉ có 1 loại kết quả). Nhập thêm cả Tài/Xỉu, Chẵn/Lẻ, Bão/Không bão để bot hoạt động chính xác!")
        return
    X_pred = df_feat.iloc[[-1]][['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll']]
    tx_proba, _ = predict_stacking(X_pred, models, 'tx')
    cl_proba, _ = predict_stacking(X_pred, models, 'cl')
    tx = "Tài" if tx_proba >= 0.5 else "Xỉu"
    cl = "Chẵn" if cl_proba >= 0.5 else "Lẻ"
    dai_diem = suggest_best_totals(df, tx)
    bao_proba, _ = predict_stacking(X_pred, models, 'bao')
    bao_pct = round(bao_proba*100,2)
    so_du_doan, dung, sai, tile = summary_stats(df)
    lines = []
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"🎯 Dự đoán: {tx} | {cl}")
    else:
        lines.append("⚠️ Dự đoán: Nên nghỉ phiên này!")
    lines.append(f"Dải điểm nên đánh: {dai_diem}")
    lines.append(f"Xác suất ra bão: {bao_pct}%")
    if max(tx_proba, 1-tx_proba) >= PROBA_ALERT:
        lines.append(f"❗️CẢNH BÁO: Xác suất {tx} vượt {int(PROBA_ALERT*100)}% – trend cực mạnh!")
    if bao_proba >= BAO_CUTOFF:
        lines.append(f"❗️CẢNH BÁO: Xác suất bão cao ({bao_pct}%) – cân nhắc vào bão!")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"Nhận định: Ưu tiên {tx}, {cl}, dải {dai_diem}. Bão {bao_pct}% – {'ưu tiên' if bao_proba >= BAO_CUTOFF else 'không nên đánh'} bão.")
    else:
        lines.append("Nhận định: Không có cửa ưu thế, nên nghỉ.")
    await update.message.reply_text('\n'.join(lines))

def main():
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
