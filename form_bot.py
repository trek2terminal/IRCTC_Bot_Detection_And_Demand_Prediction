from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import random
import string

# Function to generate a secure password (8+ chars with all requirements)
def generate_secure_password(length=8):
    if length < 8:
        raise ValueError("Password length must be at least 8")

    specials = "!@#$%^&*()_+=<>?{}[]"
    upper = random.choice(string.ascii_uppercase)
    lower = random.choice(string.ascii_lowercase)
    digit = random.choice(string.digits)
    special = random.choice(specials)

    remaining_length = length - 4
    remaining = ''.join(random.choices(
        string.ascii_letters + string.digits + specials,
        k=remaining_length
    ))

    # Shuffle to avoid fixed position pattern
    password = list(upper + lower + digit + special + remaining)
    random.shuffle(password)
    return ''.join(password)

# Generate random username
def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Configure Chrome
options = Options()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Optional: Uncomment to run without UI

# Start browser session
driver = webdriver.Chrome(options=options)

try:
    for attempt in range(1, 11):
        print(f"\n🔁 Attempt #{attempt}")
        driver.get("http://127.0.0.1:5000")
        time.sleep(2)

        username = f"user_{random_string(5)}"
        password = generate_secure_password(10)  # 10-char password (can set to 8+)

        driver.find_element(By.ID, "username").clear()
        driver.find_element(By.ID, "username").send_keys(username)
        driver.find_element(By.ID, "password").clear()
        driver.find_element(By.ID, "password").send_keys(password)

        captcha_text = driver.find_element(By.ID, "captchaText").text
        print(f"[🔐] CAPTCHA: {captcha_text}")
        driver.find_element(By.ID, "captchaInput").clear()
        driver.find_element(By.ID, "captchaInput").send_keys(captcha_text)

        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
        login_button.click()

        time.sleep(2)
        current_url = driver.current_url
        print(f"[➡️] Redirected to: {current_url}")

        if "access-denied" in current_url:
            print("🚫 Access Denied page reached. Stopping further attempts.")
            break

        time.sleep(1.5)

except Exception as e:
    print("❌ Error:", e)

finally:
    driver.quit()
