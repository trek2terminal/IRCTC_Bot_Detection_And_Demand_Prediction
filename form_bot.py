from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import random
import string

# Function to generate random username/password
def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Configure Chrome
options = Options()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Optional: Uncomment to run without UI

# Start browser session (preserves Flask session)
driver = webdriver.Chrome(options=options)

try:
    for attempt in range(1, 11):  # 5 login attempts
        print(f"\n🔁 Attempt #{attempt}")
        driver.get("http://127.0.0.1:5000")
        time.sleep(2)  # Let page load

        # Generate different credentials each time
        username = f"user_{random_string(5)}"
        password = f"pass_{random_string(6)}"

        driver.find_element(By.ID, "username").clear()
        driver.find_element(By.ID, "username").send_keys(username)
        driver.find_element(By.ID, "password").clear()
        driver.find_element(By.ID, "password").send_keys(password)

        # Read captcha
        captcha_text = driver.find_element(By.ID, "captchaText").text
        print(f"[🔐] CAPTCHA: {captcha_text}")
        driver.find_element(By.ID, "captchaInput").clear()
        driver.find_element(By.ID, "captchaInput").send_keys(captcha_text)

        # Click login
        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
        login_button.click()

        # Wait for redirect or error
        time.sleep(2)
        current_url = driver.current_url
        print(f"[➡️] Redirected to: {current_url}")

        if "access-denied" in current_url:
            print("🚫 Access Denied page reached. Stopping further attempts.")
            break

        time.sleep(1.5)  # Small delay between attempts

except Exception as e:
    print("❌ Error:", e)

finally:
    driver.quit()
