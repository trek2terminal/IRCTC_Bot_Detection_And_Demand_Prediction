from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
import pandas as pd
import os
os.makedirs("logs", exist_ok=True)

print("Starting app.py...")

# 🔹 Load the trained bot detection model
model = tf.keras.models.load_model("model/bot_detector_model.h5")

# 🔹 Map scroll behavior (text values) to numeric values for model input
scroll_map = {
    "none": 0,
    "slow": 1,
    "medium": 2,
    "fast": 3
}

# 🔹 Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong key in production

# 🔹 Home page
@app.route('/')
def index():
    return render_template("index.html")

# 🔹 Access denied page after 5 failed bot attempts
@app.route('/access-denied')
def access_denied():
    return render_template("access_denied.html")

# 🔹 Welcome page after successful login
@app.route('/welcome')
def welcome():
    username = request.args.get("username", "User")
    return render_template("welcome.html", username=username)

# 🔹 Bot prediction route
@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = request.json
        print("🟢 Received Data:", data)

        # Logging login attempt
        from datetime import datetime
        import csv
        log_file = "logs/login_attempts.csv"
        fieldnames = [
            "timestamp", "username", "mouse_movement_units", "typing_speed_cpm",
            "click_pattern_score", "time_spent_on_page_sec", "scroll_behavior",
            "captcha_success", "form_fill_time_sec"
        ]
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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

        # Prepare input for model
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

        # Predict
        prob = model.predict(sample)[0][0]
        is_bot = int(prob > 0.5)
        print(f"🔍 Prediction: {is_bot} (Prob: {prob:.4f})")

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
                    "message": f"⚠️ Unusual behavior detected. {attempts_left} attempts remaining."
                })
        else:
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


# 🔹 Run the app
if __name__ == '__main__':
    print("\nClick here: http://127.0.0.1:5000\n")
    app.run(debug=True)




