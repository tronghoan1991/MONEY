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

# ==== CONFIG ====
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
MIN_BATCH = 5
ROLLING_WINDOW = 50
PROBA_CUTOFF = 0.62
PROBA_ALERT = 0.75
BAO_CUTOFF = 0.03
TRAIN_EVERY = 5  # Chỉ train lại model khi có >=5 phiên mới

MODEL_PATH = "ml_stack.joblib"
MODEL_META = "ml_meta.txt"
SESSION_FILE = "session_time.txt"
LAST_PLAY_FILE = "last_play_time.txt"
MIN_SESSION_INPUT = 10      # Số phiên tối đa cần nhập để bắt trend mới
SESSION_BREAK_MINUTES = 30  # Bao nhiêu phút nghỉ thì bot coi là session mới

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
    # Tạo bảng nếu chưa có
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            input TEXT,
            actual TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            bot_predict TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_result(input_str, actual, bot_predict=None):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    now = datetime.now()
    if bot_predict is not None:
        cur.execute(
            "INSERT INTO history (input, actual, bot_predict, created_at) VALUES (%s, %s, %s, %s);",
            (input_str, actual, bot_predict, now)
        )
    else:
        cur.execute(
            "INSERT INTO history (input, actual, created_at) VALUES (%s, %s, %s);",
            (input_str, actual, now)
        )
    conn.commit()
    cur.close()
    conn.close()

def fetch_history(limit=10000):
    # Sử dụng SQLAlchemy để tránh warning pandas!
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT input, actual, bot_predict, created_at FROM history ORDER BY id ASC LIMIT %s" % limit,
        engine
    )
    engine.dispose()
    # Chỉ lấy các dòng input là 3 số
    df = df[df['input'].str.match(r"^\d+\s+\d+\s+\d+$", na=False)]
    return df

def delete_all_history():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE history;")
    conn.commit()
    cur.close()
    conn.close()

def make_features(df):
    df = df[df['input'].str.match(r"^\d+\s+\d+\s+\d+$", na=False)].copy()
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
            models[key] = None
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression().fit(X, y)
            rf = RandomForestClassifier(n_estimators=100).fit(X, y)
            xgbc = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss').fit(X, y)
        models[key] = (lr, rf, xgbc)
    joblib.dump(models, MODEL_PATH)
    # Lưu số lượng phiên thực tế đã train vào file meta
    with open(MODEL_META, "w") as f:
        f.write(str(len(df)))

def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def summary_stats(df):
    if 'bot_predict' in df.columns:
        df_pred = df[df['bot_predict'].notnull()]
        so_du_doan = len(df_pred)
        dung = (df_pred['bot_predict'] == df_pred['actual']).sum()
        sai = so_du_doan - dung
        tile = round((dung / so_du_doan) * 100, 2) if so_du_doan else 0
        return so_du_doan, dung, sai, tile
    else:
        return 0, 0, 0, 0

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

def predict_stacking(X_pred, models, key):
    if models[key] is None:
        return 0.5, [0.5, 0.5, 0.5]
    lr, rf, xgbc = models[key]
    prob_lr = lr.predict_proba(X_pred)[0][1]
    prob_rf = rf.predict_proba(X_pred)[0][1]
    prob_xgb = xgbc.predict_proba(X_pred)[0][1]
    probs = np.array([prob_lr, prob_rf, prob_xgb])
    return probs.mean(), probs

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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    create_table()
    # Xác nhận lệnh reset
    if user_id in PENDING_RESET and PENDING_RESET[user_id]:
        if text.upper() == "XÓA HẾT":
            delete_all_history()
            PENDING_RESET[user_id] = False
            await update.message.reply_text("✅ Đã xóa toàn bộ dữ liệu lịch sử.")
        else:
            PENDING_RESET[user_id] = False
            await update.message.reply_text("❌ Hủy thao tác xóa.")
        return

    now = datetime.now()
    last_play = load_last_play()
    session_start = load_session_start()
    need_new_session = False
    minutes_since_last = None

    # Check nếu có last_play
    if last_play:
        delta = (now - last_play).total_seconds() / 60
        minutes_since_last = int(delta)
        if delta >= SESSION_BREAK_MINUTES:
            need_new_session = True
    else:
        need_new_session = True

    # Nếu cần bắt đầu session mới tự động
    if need_new_session:
        save_session_start(now)
        if minutes_since_last:
            await update.message.reply_text(
                f"⏰ Bạn đã không chơi trong {minutes_since_last} phút. "
                f"Hãy nhập tối đa {MIN_SESSION_INPUT} phiên mới để bot bắt lại trend session!"
            )
        else:
            await update.message.reply_text(
                f"⏰ Bắt đầu session mới! Hãy nhập tối đa {MIN_SESSION_INPUT} phiên mới để bot bắt lại trend."
            )

    # Update lại thời gian chơi mới nhất
    save_last_play(now)

    m = re.match(r"^(\d{3})$", text)
    m2 = re.match(r"^(\d+)\s+(\d+)\s+(\d+)$", text)
    if not (m or m2):
        await update.message.reply_text("Vui lòng nhập kết quả theo định dạng: 456 hoặc 4 5 6.")
        return
    numbers = [int(x) for x in (m.group(1) if m else " ".join([m2.group(1), m2.group(2), m2.group(3)]))]
    input_str = f"{numbers[0]} {numbers[1]} {numbers[2]}"
    total = sum(numbers)
    actual = "Tài" if total >= 11 else "Xỉu"

    # Lấy dự đoán gần nhất (nếu có)
    df = fetch_history(10000)
    last_predict = None
    if len(df) > 0 and df.iloc[-1]['bot_predict']:
        last_predict = df.iloc[-1]['bot_predict']

    insert_result(input_str, actual, last_predict)

    # Đọc lại lịch sử thực tế
    df = fetch_history(10000)
    session_start = load_session_start()
    if session_start:
        df_session = df[df['created_at'] >= session_start]
    else:
        df_session = df
    if len(df_session) < MIN_SESSION_INPUT:
        await update.message.reply_text(f"Bạn cần nhập tối đa {MIN_SESSION_INPUT} phiên mới (sau khi bắt đầu session) để bot bắt đầu dự đoán trend session hiện tại!")
        return

    # Train model bằng toàn bộ lịch sử
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
    if (models is None or 
        models['tx'] is None or
        models['cl'] is None):
        lines = []
        lines.append(f"✔️ Đã lưu kết quả: {''.join(str(n) for n in numbers)}")
        lines.append("⚠️ Chưa đủ dữ liệu đa dạng để dự đoán (lịch sử cần đủ cả Tài/Xỉu và Chẵn/Lẻ). Nhập thêm các tổng thấp và cao, tổng chẵn/lẻ để bot hoạt động chính xác!")
        await update.message.reply_text('\n'.join(lines))
        return

    # Dự đoán rolling trend chỉ dựa trên session hiện tại
    df_feat_session = make_features(df_session)
    X_pred = df_feat_session.iloc[[-1]][['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll']]
    tx_proba, tx_probs = predict_stacking(X_pred, models, 'tx')
    tx = "Tài" if tx_proba >= 0.5 else "Xỉu"
    cl_proba, cl_probs = predict_stacking(X_pred, models, 'cl')
    cl = "Chẵn" if cl_proba >= 0.5 else "Lẻ"
    dai_diem = suggest_best_totals(df_session, tx)
    # BÃO chỉ dự đoán nếu đã có model
    bao_pct = "-"
    if models.get('bao') is not None:
        bao_proba, bao_probs = predict_stacking(X_pred, models, 'bao')
        bao_pct = round(bao_proba*100,2)
    else:
        bao_proba = None

    # Lưu lại dự đoán vào DB để so sánh đúng/sai
    insert_result("BOT_PREDICT", None, tx)

    so_du_doan, dung, sai, tile = summary_stats(fetch_history(10000))
    lines = []
    lines.append(f"✔️ Đã lưu kết quả: {''.join(str(n) for n in numbers)}")
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"🎯 Dự đoán: {tx} | {cl}")
    else:
        lines.append("⚠️ Dự đoán: Nên nghỉ phiên này!")
    lines.append(f"Dải điểm nên đánh: {dai_diem}")
    if bao_pct != "-":
        lines.append(f"Xác suất ra bão: {bao_pct}%")
        if bao_proba and bao_proba >= BAO_CUTOFF and models['bao'] is not None:
            lines.append(f"❗️CẢNH BÁO: Xác suất bão cao ({bao_pct}%) – cân nhắc vào bão!")
    else:
        lines.append(f"Chưa đủ dữ liệu để dự đoán bão.")
    if max(tx_proba, 1-tx_proba) >= PROBA_ALERT:
        lines.append(f"❗️CẢNH BÁO: Xác suất {tx} vượt {int(PROBA_ALERT*100)}% – trend cực mạnh!")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"Nhận định: Ưu tiên {tx}, {cl}, dải {dai_diem}. Bão {bao_pct}% – {'ưu tiên' if bao_proba and bao_proba >= BAO_CUTOFF and models['bao'] is not None else 'không nên đánh'} bão.")
    else:
        lines.append("Nhận định: Không có cửa ưu thế, nên nghỉ.")
    await update.message.reply_text('\n'.join(lines))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 Chào mừng đến với Sicbo ML Bot!\n\n"
        "Các lệnh hỗ trợ:\n"
        "/start – Xem hướng dẫn và danh sách lệnh\n"
        "/predict – Dự đoán phiên tiếp theo\n"
        "/stats – Thống kê hiệu suất dự đoán\n"
        "/reset – Xóa toàn bộ lịch sử data (cần xác nhận)\n"
        "Không cần nhập lệnh session nữa! Bot sẽ tự động nhận biết nếu bạn nghỉ lâu và nhắc bắt trend mới.\n\n"
        "Nhập 3 số kết quả (vd: 456 hoặc 4 5 6) để lưu và cập nhật model.\n"
        f"Nếu nghỉ quá {SESSION_BREAK_MINUTES} phút, bot sẽ tự động yêu cầu nhập tối đa {MIN_SESSION_INPUT} phiên đầu để bắt lại trend session!"
    )
    await update.message.reply_text(msg)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = fetch_history(10000)
    session_start = load_session_start()
    if session_start:
        df_session = df[df['created_at'] >= session_start]
    else:
        df_session = df
    if len(df_session) < MIN_SESSION_INPUT:
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
    if (models is None or 
        models['tx'] is None or
        models['cl'] is None):
        await update.message.reply_text("⚠️ Chưa đủ dữ liệu đa dạng để dự đoán (lịch sử cần đủ cả Tài/Xỉu và Chẵn/Lẻ). Nhập thêm các tổng thấp và cao, tổng chẵn/lẻ để bot hoạt động chính xác!")
        return
    df_feat_session = make_features(df_session)
    X_pred = df_feat_session.iloc[[-1]][['total', 'even', 'tai_roll', 'xiu_roll', 'chan_roll', 'le_roll', 'bao_roll']]
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
    insert_result("BOT_PREDICT", None, tx)
    so_du_doan, dung, sai, tile = summary_stats(fetch_history(10000))
    lines = []
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"🎯 Dự đoán: {tx} | {cl}")
    else:
        lines.append("⚠️ Dự đoán: Nên nghỉ phiên này!")
    lines.append(f"Dải điểm nên đánh: {dai_diem}")
    if bao_pct != "-":
        lines.append(f"Xác suất ra bão: {bao_pct}%")
        if bao_proba and bao_proba >= BAO_CUTOFF and models['bao'] is not None:
            lines.append(f"❗️CẢNH BÁO: Xác suất bão cao ({bao_pct}%) – cân nhắc vào bão!")
    else:
        lines.append(f"Chưa đủ dữ liệu để dự đoán bão.")
    if max(tx_proba, 1-tx_proba) >= PROBA_ALERT:
        lines.append(f"❗️CẢNH BÁO: Xác suất {tx} vượt {int(PROBA_ALERT*100)}% – trend cực mạnh!")
    lines.append(f"BOT đã dự đoán: {so_du_doan} phiên | Đúng: {dung} | Sai: {sai} | Tỉ lệ đúng: {tile}%")
    if max(tx_proba, 1-tx_proba) >= PROBA_CUTOFF:
        lines.append(f"Nhận định: Ưu tiên {tx}, {cl}, dải {dai_diem}. Bão {bao_pct}% – {'ưu tiên' if bao_proba and bao_proba >= BAO_CUTOFF and models['bao'] is not None else 'không nên đánh'} bão.")
    else:
        lines.append("Nhận định: Không có cửa ưu thế, nên nghỉ.")
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

def main():
    create_table()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
