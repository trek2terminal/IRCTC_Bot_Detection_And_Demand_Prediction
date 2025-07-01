from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
import tensorflow as tf
import pandas as pd
import os
import csv
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import random

# Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong key

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "login_attempts.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "bot_detector_model.h5")
#FONT_PATH = os.path.join(BASE_DIR, "static", "fonts", "arial.ttf")  # Font file required for CAPTCHA
FONT_PATH = os.path.join(BASE_DIR, "static", "fonts", "DejaVuSans.ttf")

# Create directories if not present
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "fonts"), exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

scroll_map = {
    "none": 0,
    "slow": 1,
    "medium": 2,
    "fast": 3
}

@app.route('/')
def index():
    return render_template("index.html")

# @app.route('/captcha-debug')
# def captcha_debug():
#     if not app.debug:
#         return jsonify({"error": "Not allowed"}), 403
#     if 'captcha_text' in session:
#         return jsonify({"captcha_text": session['captcha_text']})
#     else:
#         return jsonify({"error": "CAPTCHA not set"}), 400


@app.route('/access-denied')
def access_denied():
    return render_template("access_denied.html")

@app.route('/welcome')
def welcome():
    username = request.args.get("username", "User")
    return render_template("welcome.html", username=username)

# ✅ Secure image-based CAPTCHA route
@app.route('/captcha')
def captcha_image():
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
    captcha_text = ''.join(random.choices(chars, k=6))
    session['captcha_text'] = captcha_text

    # Create image
    img_width = 200
    img_height = 70
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 32)
    except IOError:
        return "Font file not found. Place the .ttf in static/fonts/", 500

    # Center the text
    # text_width, text_height = draw.textsize(captcha_text, font=font)
    bbox = font.getbbox(captcha_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2
    draw.text((x, y), captcha_text, font=font, fill=(0, 0, 0))

    # Add visual noise
    for _ in range(25):
        x = random.randint(0, 160)
        y = random.randint(0, 60)
        draw.point((x, y), fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/predict', methods=["POST"])
def predict():
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

        # ✅ CAPTCHA Verification
        if 'captcha_text' not in session or data.get("captcha_input", "") != session["captcha_text"]:
            return jsonify({
                "prediction": 1,
                "blocked": False,
                "message": "❌ Incorrect CAPTCHA entered."
            })

        # ✅ Logging
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "username", "mouse_movement_units", "typing_speed_cpm",
                "click_pattern_score", "time_spent_on_page_sec", "scroll_behavior",
                "captcha_success", "form_fill_time_sec"
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
                "form_fill_time_sec": data["form_fill_time_sec"]
            })

        # ✅ Data preparation
        sample = pd.DataFrame([{
            "mouse_movement_units": float(data["mouse_movement_units"]),
            "typing_speed_cpm": float(data["typing_speed_cpm"]),
            "click_pattern_score": float(data["click_pattern_score"]),
            "time_spent_on_page_sec": float(data["time_spent_on_page_sec"]),
            "scroll_behavior": scroll_map.get(data["scroll_behavior"], 0),
            "captcha_success": int(data["captcha_success"]),
            "form_fill_time_sec": float(data["form_fill_time_sec"])
        }])

        print("🧠 Prepared Data:\n", sample)
        prob = model.predict(sample)[0][0]
        print(f"🔍 Raw Probability: {prob:.4f}")

        # ✅ Bot rules
        rule_bot = (
            float(data["typing_speed_cpm"]) > 1200 or
            float(data["mouse_movement_units"]) < 30 or
            (data["scroll_behavior"] == "none" and float(data["time_spent_on_page_sec"]) < 2)
        )

        is_bot = int(prob >= 0.5 or rule_bot)

        if is_bot:
            session['bot_attempts'] = session.get('bot_attempts', 0) + 1
            attempts_left = 5 - session['bot_attempts']
            print(f"⚠️ Bot Attempt #{session['bot_attempts']} (Attempts left: {attempts_left})")

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

        # ✅ Legitimate user
        session.clear()
        return jsonify({
            "prediction": 0,
            "blocked": False,
            "redirect_url": url_for("welcome", username=data.get("username", "User"))
        })

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("\n🚀 Server Running: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
