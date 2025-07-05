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
from flask import request, render_template
import joblib
import time

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

    print("🔁 Retraining bot detection model...")

    log_csv_path = os.path.join(LOG_DIR, "retraining_history.csv")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = pd.read_csv(LOG_FILE)
        df['scroll_behavior'] = df['scroll_behavior'].astype('category').cat.codes
        df.dropna(inplace=True)

        X = df[[
            "mouse_movement_units", "typing_speed_cpm", "click_pattern_score",
            "time_spent_on_page_sec", "scroll_behavior", "captcha_success",
            "form_fill_time_sec"
        ]]
        y = df["label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

        # Save model
        new_model.save(MODEL_PATH)

        # Extract final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        # Save to CSV
        file_exists = os.path.exists(log_csv_path)
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "train_accuracy", "val_accuracy", "train_loss", "val_loss"
                ])
            writer.writerow([
                now, round(final_train_acc, 4), round(final_val_acc, 4),
                round(final_train_loss, 4), round(final_val_loss, 4)
            ])

        print(f"✅ Retraining complete. Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}")
        return True

    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        with open(os.path.join(LOG_DIR, "retraining_log.txt"), "a") as f:
            f.write(f"[{now}] ❌ Retraining failed: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        return False





# Load encoders and model once (ideally at the top of your file or in app setup)
class_encoder = joblib.load('model/class_encoder.pkl')
quota_encoder = joblib.load('model/quota_encoder.pkl')
# Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong key

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "login_attempts.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "bot_detector_model.h5")
PREDICT_MODEL_PATH = os.path.join(BASE_DIR, "model", "train_predictor_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "train_availability_data.csv")
FONT_PATH = os.path.join(BASE_DIR, "static", "fonts", "DejaVuSans.ttf")

# Create directories if not present
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "fonts"), exist_ok=True)

# Load models and data
model = tf.keras.models.load_model(MODEL_PATH)
predict_model = joblib.load(PREDICT_MODEL_PATH)
data_df = pd.read_csv(DATA_PATH)

scroll_map = {
    "none": 0,
    "slow": 1,
    "medium": 2,
    "fast": 3
}


@app.route('/')
def index():
    return render_template("index.html")

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

    # bbox = font.getbbox(captcha_text)
    # text_width = bbox[2] - bbox[0]
    # text_height = bbox[3] - bbox[1]
    # x = (img_width - text_width) // 2
    # y = (img_height - text_height) // 2
    # draw.text((x, y), captcha_text, font=font, fill=(0, 0, 0))
    #
    # for _ in range(25):
    #     x = random.randint(0, 160)
    #     y = random.randint(0, 60)
    #     draw.point((x, y), fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    #
    #
    #
    # buf = io.BytesIO()
    # img.save(buf, format='PNG')
    # buf.seek(0)
    # return send_file(buf, mimetype='image/png')

    # Optional: Blur the image
    img = img.filter(ImageFilter.GaussianBlur(1))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/predict', methods=["POST"])
def predict():
    global model
    try:
        data = request.json
        print("🟢 Received Data:", data)

        required_fields = [
            "mouse_movement_units", "typing_speed_cpm", "click_pattern_score",
            "time_spent_on_page_sec", "scroll_behavior", "captcha_success", "form_fill_time_sec",
            "captcha_input", "username"
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
            "form_fill_time_sec": float(data["form_fill_time_sec"])
        }])

        print("Prepared Data:\n", sample)
        prob = model.predict(sample)[0][0]
        print(f" Raw Probability: {prob:.4f}")

        # ✅ Extreme rule-based check
        def is_extreme_outlier(d):
            return (
                    float(d["typing_speed_cpm"]) > 800 or
                    float(d["mouse_movement_units"]) < 60 or
                    float(d["click_pattern_score"]) < 0.05 or
                    float(d["form_fill_time_sec"]) < 3 or
                    (d["scroll_behavior"] == "none" and float(d["time_spent_on_page_sec"]) < 4) or
                    int(d["captcha_success"]) == 0
            )

        captcha_time_taken = time.time() - session.get("captcha_start_time", time.time())
        print(f"⏱️ CAPTCHA solved in {captcha_time_taken:.2f} seconds")

        # More accurate decision logic
        if prob < 0.2:
            is_bot = 0  # Very confident it's human
        elif prob < 0.5:
            if is_extreme_outlier(data) or captcha_time_taken < 1.5:
                print("⚠️ Human-like bot: model trusts user, but behavior is too abnormal.")
                is_bot = 1
            else:
                is_bot = 0
        else:
            is_bot = 1  # Likely bot

        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "username", "mouse_movement_units", "typing_speed_cpm",
                "click_pattern_score", "time_spent_on_page_sec", "scroll_behavior",
                "captcha_success", "form_fill_time_sec","label"
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
                "label": is_bot  # ✅ This line is important!
            })
        try:
            df = pd.read_csv(LOG_FILE)
            if len(df) % 10 == 0:

                if retrain_bot_model():
                    model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print("⚠️ Retraining failed:", e)



        if is_bot:
            session['bot_attempts'] = session.get('bot_attempts', 0) + 1
            attempts_left = 5 - session['bot_attempts']
            print(f" Bot Attempt {session['bot_attempts']} (Attempts left: {attempts_left})")

            if session['bot_attempts'] >= 5:
                session.clear()
                return jsonify({
                    "prediction": 1,
                    "blocked": True,
                    "message": "🚫 Access Denied: Multiple bot attempts detected.",
                    "redirect_url": url_for("access_denied")
                })

            else:
                return jsonify({
                    "prediction": 1,
                    "blocked": False,
                    "attempts_left": attempts_left,
                    "message": f"⚠️ Suspicious behavior detected. {attempts_left} attempts remaining."
                })

        session.clear()
        return jsonify({
            "prediction": 0,
            "blocked": False,
            "redirect_url": url_for("welcome", username=data.get("username", "User"))
        })

    except Exception as e:
        print("Error in /predict:",str(e))
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

@app.route('/stations')
def stations():
    query = request.args.get('q', '').lower()
    results = []
    if query:
        subset = data_df[data_df['source_name'].str.lower().str.contains(query) | data_df['destination_name'].str.lower().str.contains(query)]
        stations = pd.concat([subset['source_name'], subset['destination_name']]).drop_duplicates().head(10)
        results = stations.tolist()
    return jsonify(results)

from datetime import datetime

@app.route('/search_trains', methods=['POST'])
def train_results():
    source = request.form.get("source")
    destination = request.form.get("destination")
    date_str = request.form.get("date")
    travel_class = request.form.get("travel_class")

    try:
        journey_date = datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.today()
        days_before_journey = (journey_date - today).days
        day_of_week = journey_date.weekday()  # Monday = 0
        month = journey_date.month
    except:
        return "Invalid date provided", 400

    print("🔎 Source:", source, "| Destination:", destination)
    print("🗂️ Sample from dataset:")
    print(data_df[['source_name', 'destination_name']].head(5))

    trains = data_df[
        (data_df['source_name'].str.lower() == source.lower()) &
        (data_df['destination_name'].str.lower() == destination.lower())
    ]

    print(f"✅ Matching trains found: {len(trains)}")

    results = []

    for _, row in trains.iterrows():
        try:
            print("🎯 Train row sample:", row[['train_no', 'train_name', 'class', 'quota']].to_dict())

            try:
                encoded_class = class_encoder.transform([row['class']])[0]
                encoded_quota = quota_encoder.transform([row['quota']])[0]
            except ValueError:
                print(f"🚫 Skipping train due to unknown class/quota: {row['class']}, {row['quota']}")
                continue

            prediction_input = pd.DataFrame([{
                'class_encoded': encoded_class,
                'quota_encoded': encoded_quota,
                'days_before_journey': days_before_journey,
                'day_of_week': day_of_week,
                'month': month,
                'is_festival': row['is_festival'],
                'past_avg_waitlist': row['past_avg_waitlist'],
                'past_punctuality': row['past_punctuality']
            }])

            print("📦 Prediction input:", prediction_input.to_dict(orient="records"))

            preds = predict_model.predict_proba(prediction_input)

            if hasattr(predict_model, 'classes_') and len(predict_model.classes_) == 2:
                class_index = list(predict_model.classes_).index(1)
                confirmation_prob = preds[0][class_index]
                confirmation = round(confirmation_prob * 100, 2)
                cancellation = round((1 - confirmation_prob) * 100, 2)
            else:
                pred_label = predict_model.predict(prediction_input)[0]
                confirmation = 100.0 if pred_label == 1 else 0.0
                cancellation = 0.0 if pred_label == 1 else 100.0

            result = {
                'train_info': f"{row['train_no']}/{row['train_name']} ({row['source']}-{row['destination']})",
                'punctuality_rate': f"{row.get('punctuality_rate', 90)}%",
                'cancellation_rate': f"{cancellation}%",
                'confirmation_chance': f"{confirmation}%"
            }
            results.append(result)

        except Exception as e:
            print(f"⚠️ Error during prediction: {e}")
            continue

    if not results:
        message = "🚫 No matching trains found or predictions are unavailable for the selected criteria."
    else:
        message = None

    return render_template("train_results.html", source=source, destination=destination, date=date_str, trains=results)


if __name__ == '__main__':
    print("\n🚀 Server Running: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
