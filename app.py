from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
import tensorflow as tf
import pandas as pd
import os
import csv
from datetime import datetime, timedelta, date
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFilter
import io
import random
import pandas as pd
import joblib
import calendar
from datetime import datetime, timedelta
import time
from io import BytesIO
import base64
import string


def retrain_bot_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import traceback
    import datetime
    import csv
    import shutil
    import joblib
    import os
    import pandas as pd
    import tensorflow as tf

    print("🔁 Retraining bot detection model...")

    log_csv_path = os.path.join(LOG_DIR, "retraining_history.csv")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = pd.read_csv(LOG_FILE)
        df['scroll_behavior'] = df['scroll_behavior'].astype('category').cat.codes
        df.dropna(inplace=True)

        # ✅ Validation checks
        label_counts = df['label'].value_counts()
        if len(df) < 100 or label_counts.min() < 10:
            print(f"⚠️ Skipping retraining due to insufficient or imbalanced data. Samples: {len(df)}, Label distribution: {label_counts.to_dict()}")

            # Log skipped attempt
            file_exists = os.path.exists(log_csv_path)
            with open(log_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "train_accuracy", "val_accuracy", "train_loss", "val_loss",
                        "old_val_accuracy", "old_val_loss", "model_updated", "comment"
                    ])
                writer.writerow([
                    now, "", "", "", "", "", "", "No", f"Insufficient data: {len(df)} samples"
                ])
            return False

        X = df[[
            "mouse_movement_units", "typing_speed_cpm", "click_pattern_score",
            "time_spent_on_page_sec", "scroll_behavior", "captcha_success",
            "form_fill_time_sec", "captcha_time_sec"
        ]]
        y = df["label"]

        # Create new scaler and scale the data
        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # ✅ Define new model
        new_model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        new_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        history = new_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ],
            verbose=0
        )

        # Extract new model metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        # ✅ Evaluate old model
        old_model = tf.keras.models.load_model(MODEL_PATH)
        old_scaler = joblib.load(os.path.join(BASE_DIR, "model", "bot_scaler.pkl"))
        X_val_old = old_scaler.transform(X_val)
        old_loss, old_acc = old_model.evaluate(X_val_old, y_val, verbose=0)

        print(f"📊 Old Model - Val Accuracy: {old_acc:.4f}, Val Loss: {old_loss:.4f}")
        print(f"📊 New Model - Val Accuracy: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}")

        # ✅ Save retraining log
        file_exists = os.path.exists(log_csv_path)
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "train_accuracy", "val_accuracy", "train_loss", "val_loss",
                    "old_val_accuracy", "old_val_loss", "model_updated", "comment"
                ])

            if final_val_acc > old_acc and final_val_loss < old_loss:
                # Backup current model
                backup_path = MODEL_PATH.replace(".h5", "_backup.h5")
                shutil.copy(MODEL_PATH, backup_path)
                print(f"🗂️ Old model backed up at: {backup_path}")

                # Save new model
                new_model.save(MODEL_PATH)
                print("✅ New model is better. Model updated.")

                # Save new scaler
                joblib.dump(new_scaler, os.path.join(BASE_DIR, "model", "bot_scaler.pkl"))
                print("✅ New scaler saved.")

                writer.writerow([
                    now, round(final_train_acc, 4), round(final_val_acc, 4),
                    round(final_train_loss, 4), round(final_val_loss, 4),
                    round(old_acc, 4), round(old_loss, 4), "Yes", "Better model"
                ])
            else:
                print("⚠️ New model did not improve performance. Keeping existing model and scaler.")
                writer.writerow([
                    now, round(final_train_acc, 4), round(final_val_acc, 4),
                    round(final_train_loss, 4), round(final_val_loss, 4),
                    round(old_acc, 4), round(old_loss, 4), "No", "Worse or equal model"
                ])
                return False

        return True

    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        with open(os.path.join(LOG_DIR, "retraining_log.txt"), "a") as f:
            f.write(f"[{now}] ❌ Retraining failed: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        return False


def get_behavioral_features(request_form):
    import json
    typing_stats = json.loads(request_form.get('typingStats', '{}'))
    interaction_flags = json.loads(request_form.get('interactionFlags', '{}'))

    features = []

    # 1. Average typing speed
    typing_times = typing_stats.get("typingSpeed", [])
    if typing_times:
        avg_typing_speed = sum(typing_times) / len(typing_times)
    else:
        avg_typing_speed = 0
    features.append(avg_typing_speed)

    # 2. Typing variance
    if len(typing_times) > 1:
        variance = sum((x - avg_typing_speed) ** 2 for x in typing_times) / len(typing_times)
    else:
        variance = 0
    features.append(variance)

    # 3. Total idle time
    features.append(typing_stats.get("totalIdleTime", 0))

    # 4. Scroll count
    features.append(interaction_flags.get("scrollCount", 0))

    # 5. Mouse move count
    features.append(interaction_flags.get("mouseMoveCount", 0))

    # 6. Focus change count
    features.append(interaction_flags.get("focusChangeCount", 0))

    # 7. Copy-paste flag
    features.append(int(interaction_flags.get("copyPasteUsed", False)))

    # 8. DOM injection flag
    features.append(int(interaction_flags.get("domInjected", False)))

    return features

def generate_captcha():
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    session['captcha_text'] = captcha_text

    img = Image.new('RGB', (150, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except:
        font = ImageFont.load_default()

    draw.text((10, 5), captcha_text, font=font, fill=(0, 0, 0))

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return f"data:image/png;base64,{image_base64}"





# Load encoders and model once (ideally at the top of your file or in app setup)

scaler = joblib.load('model/bot_scaler.pkl')  # ✅ Load scaler for inference
# Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong key

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "login_attempts.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "bot_detector_model.h5")

DATA_PATH = os.path.join(BASE_DIR, "data", "train_availability_data.csv")
FONT_PATH = os.path.join(BASE_DIR, "static", "fonts", "DejaVuSans.ttf")
# CSVs
ticket_df = pd.read_csv('train_search_dataset.csv')          # For predictions & filtering
train_display_df = pd.read_csv('train_search_dataset.csv')  # For station name suggestions

# Create directories if not present
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "fonts"), exist_ok=True)

# Load models and data
model = tf.keras.models.load_model(MODEL_PATH)
# predict_model = joblib.load(PREDICT_MODEL_PATH)
data_df = pd.read_csv(DATA_PATH)

scroll_map = {
    "none": 0,
    "slow": 1,
    "medium": 2,
    "fast": 3
}

# 🔒 Hybrid login blocker rule (place here)
def is_hybrid_login(data):
    suspicious_signals = 0

    # Mouse movement: too little = suspicious
    if float(data["mouse_movement_units"]) < 500:
        suspicious_signals += 1

    # Typing speed: very fast or very slow is suspicious (normal ~120-250 CPM)
    typing_speed = float(data["typing_speed_cpm"])
    if typing_speed < 80 or typing_speed > 300:
        suspicious_signals += 1

    # Click pattern score: too low = bot-like
    if float(data["click_pattern_score"]) < 0.005:
        suspicious_signals += 1

    # No scrolling is slightly suspicious
    if data["scroll_behavior"] == "none":
        suspicious_signals += 1

    # Form fill time: too fast = suspicious
    if float(data["form_fill_time_sec"]) < 5:
        suspicious_signals += 1

    # CAPTCHA pass reduces suspicion strongly
    if int(data["captcha_success"]) == 1:
        suspicious_signals -= 2

    # If 3 or more signs, treat as hybrid/bot
    return suspicious_signals >= 3





@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        captcha_input = request.form['captcha']
        captcha_text = session.get('captcha_text', '')

        # CAPTCHA verification
        if captcha_input.strip().lower() != captcha_text.lower():
            return redirect(url_for('access_denied'))

        # Extract behavioral features
        features = get_behavioral_features(request.form)

        # Predict using the loaded bot detection model
        prediction = bot_model.predict([features])[0]

        if prediction == "human":
            session['username'] = username
            return redirect(url_for('welcome'))
        else:
            return redirect(url_for('access_denied'))

    return render_template('index.html', captcha_image=generate_captcha())


@app.route('/captcha-debug')
def captcha_debug():
    if not app.debug:
        return jsonify({"error": "Not allowed"}), 403
    if 'captcha_text' in session:
        return jsonify({"captcha_text": session['captcha_text']})
    else:
        return jsonify({"error": "CAPTCHA not set"}), 400

@app.route('/access-denied')
def access_denied():
    return render_template("access_denied.html")


@app.route('/welcome')
def welcome():
    username = request.args.get("username", "User")
    return render_template(
        "welcome.html",
        username=username,
        date=date,  # ✅ pass `date` to template
        timedelta=timedelta  # ✅ pass `timedelta` to template
    )


@app.route('/captcha')
def captcha_image():
    session['captcha_start_time'] = time.time()
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
    captcha_text = ''.join(random.choices(chars, k=6))
    session['captcha_text'] = captcha_text

    img_width = 200
    img_height = 70
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 32)
    except IOError:
        return "Font file not found. Place the .ttf in static/fonts/", 500

        # Add background noise: lines
    for _ in range(10):
        x1 = random.randint(0, img_width)
        y1 = random.randint(0, img_height)
        x2 = random.randint(0, img_width)
        y2 = random.randint(0, img_height)
        draw.line(((x1, y1), (x2, y2)),
                  fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)), width=2)

        # Draw each character with random rotation
    for i, char in enumerate(captcha_text):
        char_img = Image.new('RGBA', (40, 60), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((5, 5), char, font=font, fill=(0, 0, 0))
        rotated = char_img.rotate(random.randint(-30, 30), resample=Image.BICUBIC, expand=1)
        img.paste(rotated, (30 * i + 10, 5), rotated)

        # Add dots
        for _ in range(300):
            x = random.randint(0, img_width - 1)
            y = random.randint(0, img_height - 1)
            draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


    # Optional: Blur the image
    img = img.filter(ImageFilter.GaussianBlur(1))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Helper function for extreme cases
def is_extreme_outlier(d):
    try:
        return (
                float(d["typing_speed_cpm"]) > 800 or
                float(d["mouse_movement_units"]) < 60 or
                float(d["click_pattern_score"]) < 0.005 or
                float(d["form_fill_time_sec"]) < 3 or
                (d["scroll_behavior"] == "none" and float(d["time_spent_on_page_sec"]) < 4) or
                int(d["captcha_success"]) == 0
        )
    except Exception as e:
        print("⚠️ Error in extreme outlier check:", e)
        return False


@app.route('/predict', methods=["POST"])
def predict():
    global model

    #  Check if the user has been permanently blocked
    if session.get('permanently_blocked'):
        return jsonify({
            "prediction": None,
            "blocked": True,
            "message": "🚫 Access Denied: You have been permanently blocked.",
            "redirect_url": url_for("access_denied")
        })

    try:
        data = request.json
        print("🟢 Received Data:", data)

        if is_hybrid_login(data):
            print("❌ Hybrid login detected. Blocking.")
            session.clear()
            return jsonify({
                "prediction": 1,
                "blocked": True,
                "message": "🚫 Access Denied: Detected suspicious hybrid login pattern.",
                "redirect_url": url_for("access_denied")
            })

        required_fields = [
            "mouse_movement_units", "typing_speed_cpm", "click_pattern_score",
            "time_spent_on_page_sec", "scroll_behavior", "captcha_success", "form_fill_time_sec",
            "captcha_input", "username", "captcha_time_sec"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        if 'captcha_text' not in session or data.get("captcha_input", "") != session["captcha_text"]:
            return jsonify({
                "prediction": 1,
                "blocked": False,
                "message": "❌ Incorrect CAPTCHA entered."
            })

        sample = pd.DataFrame([{
            "mouse_movement_units": float(data["mouse_movement_units"]),
            "typing_speed_cpm": float(data["typing_speed_cpm"]),
            "click_pattern_score": float(data["click_pattern_score"]),
            "time_spent_on_page_sec": float(data["time_spent_on_page_sec"]),
            "scroll_behavior": scroll_map.get(data["scroll_behavior"], 0),
            "captcha_success": int(data["captcha_success"]),
            "form_fill_time_sec": float(data["form_fill_time_sec"]),
            "captcha_time_sec": float(data["captcha_time_sec"])
        }])

        print("Prepared Raw Data:\n", sample)

        bot_scaler = joblib.load(os.path.join(BASE_DIR, "model", "bot_scaler.pkl"))
        sample_scaled = bot_scaler.transform(sample)

        prob = model.predict(sample_scaled)[0][0]
        print(f"Scaled Probability: {prob:.4f}")

        captcha_time_taken = float(data.get("captcha_time_sec", 0))
        print(f"⏱️ CAPTCHA solved in {captcha_time_taken:.2f} seconds")



        # More aggressive hybrid detection
        if prob < 0.3:
            is_bot = 0
        elif prob < 0.7:
            if is_extreme_outlier(data) or captcha_time_taken < 3:
                is_bot = 1
            else:
                is_bot = 1 if int(data["captcha_success"]) == 1 and float(data["form_fill_time_sec"]) < 4 else 0
        else:
            is_bot = 1

        # Debug log
        with open("logs/debug_predictions.log", "a") as debug_log:
            debug_log.write(f"\n=== Prediction Log {datetime.now()} ===\n")
            debug_log.write(f"Input Data:\n{sample.to_dict(orient='records')}\n")
            debug_log.write(f"Model Probability: {prob:.4f}\n")
            debug_log.write(f"Classified as bot: {is_bot}\n")

        # Save input with label
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "username", "mouse_movement_units", "typing_speed_cpm",
                "click_pattern_score", "time_spent_on_page_sec", "scroll_behavior",
                "captcha_success", "form_fill_time_sec", "captcha_time_sec", "label"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": data.get("username", "unknown"),
                "mouse_movement_units": data["mouse_movement_units"],
                "typing_speed_cpm": data["typing_speed_cpm"],
                "click_pattern_score": data["click_pattern_score"],
                "time_spent_on_page_sec": data["time_spent_on_page_sec"],
                "scroll_behavior": data["scroll_behavior"],
                "captcha_success": data["captcha_success"],
                "form_fill_time_sec": data["form_fill_time_sec"],
                "captcha_time_sec": data["captcha_time_sec"],
                "label": is_bot
            })

        try:
            df = pd.read_csv(LOG_FILE)
            if len(df) % 10 == 0:
                if retrain_bot_model():
                    model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print("⚠️ Retraining failed:", e)

        # If suspicious bot-human hybrid — treat as bot and count attempt
        if is_bot and int(data["captcha_success"]) == 1:
            print("🚨 Hybrid bot-human behavior detected.")

            session['bot_attempts'] = session.get('bot_attempts', 0) + 1
            attempts_left = 5 - session['bot_attempts']
            print(f"Hybrid Attempt {session['bot_attempts']} (Attempts left: {attempts_left})")

            if session['bot_attempts'] >= 5:
                session['permanently_blocked'] = True
                return jsonify({
                    "prediction": 1,
                    "blocked": True,
                    "message": "🚫 Access Denied: Multiple hybrid attempts detected.",
                    "redirect_url": url_for("access_denied")
                })

            return jsonify({
                "prediction": 1,
                "blocked": False,
                "attempts_left": attempts_left,
                "message": f"⚠️ Suspicious hybrid login behavior. {attempts_left} attempts remaining."
            })

        # Bot with failed CAPTCHA
        if is_bot:
            session['bot_attempts'] = session.get('bot_attempts', 0) + 1
            attempts_left = 5 - session['bot_attempts']
            print(f" Bot Attempt {session['bot_attempts']} (Attempts left: {attempts_left})")

            if session['bot_attempts'] >= 5:
                session['permanently_blocked'] = True
                return jsonify({
                    "prediction": is_bot,
                    "blocked": True,
                    "message": "🚫 Access Denied: Multiple suspicious attempts detected.",
                    "redirect_url": url_for("access_denied")
                })

            else:
                return jsonify({
                    "prediction": 1,
                    "blocked": False,
                    "attempts_left": attempts_left,
                    "message": f"⚠️ Suspicious behavior detected. {attempts_left} attempts remaining."
                })

        # ✅ Human - allow access
        # ✅ Human detected — reset bot attempts
        session.pop('bot_attempts', None)
        return jsonify({
            "prediction": 0,
            "blocked": False,
            "redirect_url": url_for("welcome", username=data.get("username", "User"))
        })


    except Exception as e:
        print("Error in /predict:", str(e))
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500



@app.route('/flag_dom_injection', methods=['POST'])
def flag_dom_injection():
    print("⚠️ DOM injection attempt detected from:", request.remote_addr)
    return '', 204


@app.route('/stations')
def stations():
    query = request.args.get('q', '').lower()
    results = []
    if query:
        subset = data_df[data_df['source_name'].str.lower().str.contains(query) | data_df['destination_name'].str.lower().str.contains(query)]
        stations = pd.concat([subset['source_name'], subset['destination_name']]).drop_duplicates().head(10)
        results = stations.tolist()
    return jsonify(results)


@app.route('/get_trains', methods=['POST'])
def get_trains():
    import pandas as pd
    from datetime import datetime

    try:
        print("\n📥 Step 1: Reading train_search_dataset.csv")
        df = pd.read_csv('train_search_dataset.csv')
        print(f"✅ Loaded train search dataset with {len(df)} rows")
        print(f"[DEBUG] Columns: {list(df.columns)}")

        # --- Step 2: Receive and parse form input ---
        source_input = request.form.get("source", "").strip()
        destination_input = request.form.get("destination", "").strip()
        journey_date_str = request.form.get("journey_date", "").strip()
        selected_class = request.form.get("class", "").strip()
        print(f"📥 Step 2: User input -> source='{source_input}', destination='{destination_input}', date='{journey_date_str}', class='{selected_class}'")

        # --- Step 3: Parse journey date ---
        journey_date = datetime.strptime(journey_date_str, "%Y-%m-%d").date()
        today = datetime.today().date()
        days_before_journey = (journey_date - today).days
        weekday = journey_date.weekday()
        month = journey_date.month
        print(f"📅 Step 3: Parsed date -> {journey_date} (in {days_before_journey} days), weekday={weekday}, month={month}")

        # --- Step 4: Build fast station lookup (deduplicated) ---
        print("🔁 Step 4: Building station lookup from dataset (fast mapping)")
        src_df = df[['source_code', 'source_name']].drop_duplicates().rename(columns={'source_code':'code','source_name':'name'})
        dst_df = df[['destination_code', 'destination_name']].drop_duplicates().rename(columns={'destination_code':'code','destination_name':'name'})
        stations_df = pd.concat([src_df, dst_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

        name_to_code = {}
        code_to_name = {}
        for _, r in stations_df.iterrows():
            name = str(r['name']).strip().lower()
            code = str(r['code']).strip().upper()
            if name and code:
                if name not in name_to_code:
                    name_to_code[name] = code
                # prefer the first seen name for a code
                if code not in code_to_name:
                    code_to_name[code] = r['name']

        print(f"[DEBUG] Station lookup built: {len(name_to_code)} names, {len(code_to_name)} codes")

        # Helper: map user input (name or code) -> station code
        def map_to_code(user_input):
            if not user_input:
                return None
            v = user_input.strip().lower()
            # exact name match
            if v in name_to_code:
                code = name_to_code[v]
                print(f"🔄 Mapped by exact name '{user_input}' -> {code}")
                return code
            # exact code match
            up = user_input.strip().upper()
            if up in code_to_name:
                print(f"🔄 Mapped by exact code '{user_input}' -> {up}")
                return up
            # fuzzy: startswith on names
            for nm, cd in name_to_code.items():
                if nm.startswith(v):
                    print(f"⚠️ Fuzzy matched '{user_input}' -> {cd} (name='{nm}')")
                    return cd
            # not found
            print(f"❌ No station match for '{user_input}'")
            return None

        mapped_source = map_to_code(source_input)
        mapped_destination = map_to_code(destination_input)

        if not mapped_source or not mapped_destination:
            print(f"❌ Invalid source/destination mapping -> source: {mapped_source}, destination: {mapped_destination}")
            return jsonify({"error": "Source or Destination not found"}), 400

        print(f"✅ Mapped source_code: {mapped_source}, destination_code: {mapped_destination}")

        # --- Step 5: Map class names (keep old mapping) ---
        class_map = {'3AC': '3A', '2AC': '2A', 'Sleeper': 'SL', '1AC': '1A', 'CC': 'CC', '2S': '2S'}
        mapped_class = class_map.get(selected_class, selected_class) if selected_class else ""
        print(f"🎫 Step 5: Mapped class input '{selected_class}' -> '{mapped_class}'")

        # --- Step 6: Hard filter (route [+ class if provided]) ---
        mask = (
            (df['source_code'] == mapped_source) &
            (df['destination_code'] == mapped_destination)
        )
        if mapped_class:
            mask = mask & (df['class'] == mapped_class)

        results = df[mask].copy()  # .copy() to avoid SettingWithCopyWarning
        print(f"🔎 Step 6: After hard filters -> {len(results)} rows")

        # Fallback: if class filtered everything out but route exists, show route-only and log
        if results.empty and mapped_class:
            route_only = df[
                (df['source_code'] == mapped_source) &
                (df['destination_code'] == mapped_destination)
            ].copy()
            if not route_only.empty:
                print("⚠️ No matches in selected class. Falling back to route-only results (showing all classes).")
                results = route_only
            else:
                print("❌ No trains found for this route (even ignoring class).")
                # Prepare empty train list for frontend results page and always redirect
                train_list = []
                session['source'] = source_input
                session['destination'] = destination_input
                session['journey_date'] = journey_date_str
                session['travel_class'] = selected_class
                session['filtered_trains'] = train_list
                return redirect('/train_results')

        # --- Step 7: Soft scoring (non-eliminating) ---
        results['day_match'] = results['weekday'] == weekday
        results['month_match'] = results['month'] == month
        results['days_diff'] = (results['days_before_journey'] - days_before_journey).abs()
        results = results.sort_values(
            by=['day_match', 'month_match', 'days_diff', 'avg_waitlist'],
            ascending=[False, False, True, True]
        )
        print(f"🔎 Step 7: Sorted results. Top rows: {min(5, len(results))}")

        # --- Step 8: Prepare final JSON-ready list (and parse confirmation percentage) ---
        train_list = []
        parse_errors = 0
        for _, row in results.iterrows():
            # parse confirmation_chance (supports "59.24%" or numeric)
            conf_val = None
            if 'confirmation_chance' in row and pd.notna(row['confirmation_chance']):
                raw = row['confirmation_chance']
                try:
                    if isinstance(raw, str):
                        raw_clean = raw.strip().replace('%', '')
                        conf_val = float(raw_clean)
                    else:
                        conf_val = float(raw)
                except Exception:
                    parse_errors += 1
                    conf_val = None
            elif 'confirmed' in row and pd.notna(row['confirmed']):
                try:
                    conf_val = float(row['confirmed']) * 100.0
                except Exception:
                    parse_errors += 1
                    conf_val = None

            # fallback if parsing failed
            if conf_val is None:
                conf_val = 0.0

            train_list.append({
                "train_no": row.get("train_no"),
                "train_name": row.get("train_name"),
                "class": row.get("class"),
                "confirmation_chance": round(conf_val, 2),                 # numeric percent (0-100)
                "confirmation_chance_text": f"{round(conf_val, 2)}%",     # display-friendly
                "punctuality_rate": row.get("punctuality_rate"),
                "avg_waitlist": int(row.get("avg_waitlist")) if pd.notna(row.get("avg_waitlist")) else None,
                "days_before_journey": int(row.get("days_before_journey")) if pd.notna(row.get("days_before_journey")) else None,
                "source_code": row.get("source_code"),
                "destination_code": row.get("destination_code")
            })

        if parse_errors:
            print(f"[WARN] {parse_errors} confirmation_chance parse errors (set to 0.0)")

        # --- Step 9: Save to session and return ---
        session['source'] = source_input
        session['destination'] = destination_input
        session['journey_date'] = journey_date_str
        session['travel_class'] = selected_class
        session['filtered_trains'] = train_list

        print(f"🎯 Final: {len(train_list)} trains prepared for frontend")
        if len(train_list) > 0:
            print(f"[DEBUG] First 5 trains: {train_list[:5]}")
        else:
            print("[DEBUG] No trains to show.")

        return redirect('/train_results')

    except Exception as e:
        print(f"🔥 Error in get_trains: {e}")
        return jsonify({"error": "Something went wrong"}), 500


@app.route('/train_results')
def show_train_results():
    trains = session.get('filtered_trains', [])
    source = session.get('source')
    destination = session.get('destination')
    journey_date = session.get('journey_date')
    travel_class = session.get('travel_class')

    return render_template('train_results.html',
                           trains=trains,
                           source=source,
                           destination=destination,
                           journey_date=journey_date,
                           travel_class=travel_class)



@app.route('/get_station_suggestions', methods=['GET'])
def get_station_suggestions():
    query = request.args.get('q', '').lower()

    # Combine source and destination names
    all_stations = pd.concat([
        train_display_df[['source_name']].rename(columns={'source_name': 'station'}),
        train_display_df[['destination_name']].rename(columns={'destination_name': 'station'})
    ])

    # Drop duplicates and nulls
    all_stations = all_stations.drop_duplicates().dropna()

    # Filter by query
    suggestions = all_stations[all_stations['station'].str.lower().str.contains(query)]

    # Return top 10
    return jsonify(suggestions['station'].drop_duplicates().head(10).tolist())

@app.route('/demand_heatmap', methods=['GET'])
def demand_heatmap():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import io, base64
    import pandas as pd
    from flask import request, jsonify
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.font_manager as fm
    import os

    # Load Poppins from static/fonts
    poppins_path = os.path.join('static', 'fonts', 'Poppins-Regular.ttf')
    poppins_bold_path = os.path.join('static', 'fonts', 'Poppins-Bold.ttf')
    fm.fontManager.addfont(poppins_path)
    fm.fontManager.addfont(poppins_bold_path)
    plt.rcParams['font.family'] = 'Poppins'

    train_no = request.args.get('train_no')
    if not train_no:
        return jsonify({'success': False, 'error': 'No train number provided.'})

    try:
        train_no = int(train_no)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid train number format.'})

    df = ticket_df[ticket_df['train_no'] == train_no].copy()
    if df.empty:
        return jsonify({'success': False, 'error': 'No data available for this train.'})

    if df['confirmation_chance'].dtype == object:
        df['confirmation_chance'] = (
            df['confirmation_chance'].astype(str).str.replace('%', '', regex=False).astype(float)
        )

    max_day = df['days_before_journey'].max()
    high = max(90, int(max_day))
    bins = [-1, 15, 30, 60, 90, high + 1]
    labels = ['<15', '15-30', '31-60', '61-90', f'>{high}']
    df['days_group'] = pd.cut(df['days_before_journey'], bins=bins, labels=labels)

    bar_data = df.groupby('days_group')['confirmation_chance'].mean().reindex(labels)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#f5f7fa')
    ax.set_facecolor('#ffffff')

    cmap = LinearSegmentedColormap.from_list("demand", ["#2ECC71", "#F1C40F", "#E74C3C"])
    colors = [cmap(v/100) for v in bar_data.values]

    for i, val in enumerate(bar_data.values):
        ax.bar(i, val, color='gray', width=0.55, alpha=0.15, zorder=0)

    bars = ax.bar(bar_data.index.astype(str), bar_data.values,
                  color=colors, edgecolor='none', zorder=2)

    ax.set_ylim(0, 100)
    ax.set_xlabel("Days Before Journey", fontsize=14, labelpad=12, color="#444")
    ax.set_ylabel("Ticket Confirmation Chance (%)", fontsize=14, labelpad=12, color="#444")

    ax.set_title(" Ticket Demand & Confirmation Trends",
                 fontsize=20, fontweight='bold', color="#222", pad=25)
    fig.text(0.5, 0.92,
             " Tip: Book earlier for higher chances — Green = Easiest, Red = Busiest",
             ha='center', fontsize=12, color="#555", style='italic')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2,
                f"{yval:.1f}%",
                ha="center", va="bottom",
                fontsize=12, fontweight='bold', color="#222",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    best_idx = bar_data.values.argmax()
    ax.annotate(
        '✅ Best Window\n(Book well in advance)',
        xy=(best_idx, bar_data.iloc[best_idx]),
        xytext=(best_idx, bar_data.max() + 15),
        arrowprops=dict(facecolor='#27AE60', edgecolor='none',
                        arrowstyle="wedge,tail_width=0.5", shrinkA=0, shrinkB=5),
        fontsize=12, color='#27AE60',
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#27AE60", lw=1.2)
    )

    legend_labels = [' Easy Booking', ' Medium Demand', ' High Demand']
    legend_colors = ['#2ECC71', '#F1C40F', '#E74C3C']
    for label, color in zip(legend_labels, legend_colors):
        ax.bar(0, 0, color=color, label=label)
    legend = ax.legend(title="Legend", loc="upper right", frameon=True,
                       facecolor='white', edgecolor='#ddd')
    legend.get_frame().set_boxstyle('round,pad=0.4')

    ax.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return jsonify({'success': True, 'img_data': img_base64})



if __name__ == '__main__':
    print("\n🚀 Server Running: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
