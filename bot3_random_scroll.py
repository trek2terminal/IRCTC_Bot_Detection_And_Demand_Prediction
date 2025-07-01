from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import random
import string

# Function to generate a secure password
def generate_secure_password(length=10):
    if length < 8:
        raise ValueError("Password must be at least 8 characters")

    specials = "!@#$%^&*()_+=<>?{}[]"
    components = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
        random.choice(specials),
    ]
    components += random.choices(string.ascii_letters + string.digits + specials, k=length - 4)
    random.shuffle(components)
    return ''.join(components)

# Random string generator
def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Simulate human-like typing
def human_type(element, text, min_delay=0.1, max_delay=0.3):
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(min_delay, max_delay))

# Simulate thinking time
def human_delay(min_time=0.8, max_time=2.5):
    time.sleep(random.uniform(min_time, max_time))

# Configure Chrome
options = Options()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Optional headless run

# Start browser
driver = webdriver.Chrome(options=options)

try:
    for attempt in range(1, 11):
        print(f"\n🔁 Attempt #{attempt}")
        driver.get("http://127.0.0.1:5000")
        human_delay(1.5, 3)

        username = f"user_{random_string(5)}"
        password = generate_secure_password(10)

        username_field = driver.find_element(By.ID, "username")
        username_field.clear()
        human_type(username_field, username)

        human_delay(0.5, 1)

        password_field = driver.find_element(By.ID, "password")
        password_field.clear()
        human_type(password_field, password)

        human_delay(0.5, 1)

        captcha_text = driver.find_element(By.ID, "captchaText").text
        print(f"[🔐] CAPTCHA: {captcha_text}")

        captcha_input = driver.find_element(By.ID, "captchaInput")
        captcha_input.clear()
        human_type(captcha_input, captcha_text, min_delay=0.2, max_delay=0.4)

        human_delay(1, 2)

        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
        login_button.click()

        time.sleep(2)
        current_url = driver.current_url
        print(f"[➡️] Redirected to: {current_url}")

        if "access-denied" in current_url:
            print("🚫 Access Denied. Stopping simulation.")
            break

        human_delay(1, 2)

except Exception as e:
    print("❌ Error:", e)

finally:
    driver.quit()
