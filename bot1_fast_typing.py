# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# import time
#
# # Configure Chrome
# options = Options()
# options.add_argument("--start-maximized")
# # options.add_argument("--headless")  # Uncomment to run in headless mode
#
# # Start persistent browser session
# driver = webdriver.Chrome(options=options)
#
# try:
#     for attempt in range(1, 7):  # 6 login attempts
#         print(f"\n🔁 [Attempt #{attempt}]")
#         driver.get("http://127.0.0.1:5000")
#         time.sleep(1.5)  # Allow page to load
#
#         # Fill credentials
#         driver.find_element(By.ID, "username").clear()
#         driver.find_element(By.ID, "username").send_keys("bot1")
#         driver.find_element(By.ID, "password").clear()
#         driver.find_element(By.ID, "password").send_keys("fast123")
#
#         # Read and enter CAPTCHA
#         captcha_text = driver.find_element(By.ID, "captchaText").text
#         print(f"[🔐] CAPTCHA: {captcha_text}")
#         driver.find_element(By.ID, "captchaInput").clear()
#         driver.find_element(By.ID, "captchaInput").send_keys(captcha_text)
#
#         # Submit form
#         driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]").click()
#
#         # Wait for navigation or error display
#         time.sleep(2)
#         current_url = driver.current_url
#         print(f"[➡️] Redirected to: {current_url}")
#
#         # Stop if blocked
#         if "access-denied" in current_url:
#             print("🚫 Access Denied page reached. Stopping further attempts.")
#             break
#
#         time.sleep(1.5)  # Optional delay before next attempt
#
# except Exception as e:
#     print("❌ Error:", e)
#
# finally:
#     driver.quit()
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import time
import random
import string
import requests

# === Configuration ===
URL = "http://127.0.0.1:5000"
HEADLESS = False  # Set to True to run without opening the browser
MAX_ATTEMPTS = 10
BEHAVIORS = ["human", "bot", "human_like_bot"]

# === Random Generators ===
def generate_secure_password(length=10):
    specials = "!@#$%^&*()_+=<>?{}[]"
    upper = random.choice(string.ascii_uppercase)
    lower = random.choice(string.ascii_lowercase)
    digit = random.choice(string.digits)
    special = random.choice(specials)
    remaining = ''.join(random.choices(string.ascii_letters + string.digits + specials, k=length - 4))
    password = list(upper + lower + digit + special + remaining)
    random.shuffle(password)
    return ''.join(password)

def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# === Browser Setup ===
options = Options()
options.add_argument("--start-maximized")
if HEADLESS:
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# === Test Loop ===
try:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        label = random.choice(BEHAVIORS)
        print(f"\n🔁 Attempt #{attempt} | Behavior: {label.upper()}")
        driver.get(URL)
        wait.until(EC.presence_of_element_located((By.ID, "username")))

        username = f"user_{random_string(5)}"
        password = generate_secure_password()

        # Simulated behavior per label
        if label == "human":
            scroll_behavior = random.choice(["medium", "slow"])
            typing_delay = random.uniform(0.1, 0.25)
            time_spent = random.uniform(5, 15)
        elif label == "bot":
            scroll_behavior = "none"
            typing_delay = random.uniform(0.01, 0.03)
            time_spent = random.uniform(1, 3)
        elif label == "human_like_bot":
            scroll_behavior = "medium"
            typing_delay = random.uniform(0.07, 0.15)
            time_spent = random.uniform(4, 6)

        # Scroll simulation
        scroll_script = {
            "slow": "window.scrollTo({top: 100, behavior: 'smooth'});",
            "medium": "window.scrollTo({top: 300, behavior: 'smooth'});",
            "fast": "window.scrollTo({top: 600, behavior: 'smooth'});",
            "none": ""
        }
        if scroll_script[scroll_behavior]:
            driver.execute_script(scroll_script[scroll_behavior])
            time.sleep(1)

        # Typing simulation
        username_input = driver.find_element(By.ID, "username")
        username_input.clear()
        for char in username:
            username_input.send_keys(char)
            time.sleep(typing_delay)

        password_input = driver.find_element(By.ID, "password")
        password_input.clear()
        for char in password:
            password_input.send_keys(char)
            time.sleep(typing_delay)

        time.sleep(1.2)  # Let session get CAPTCHA

        # Fetch session cookies
        cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
        captcha_response = requests.get(f"{URL}/captcha-debug", cookies=cookies)
        captcha_text = captcha_response.json().get("captcha_text", "")

        print(f"[🔐] CAPTCHA Text: {captcha_text}")

        driver.find_element(By.ID, "captchaInput").clear()
        driver.find_element(By.ID, "captchaInput").send_keys(captcha_text)

        time.sleep(time_spent)

        # Submit form
        driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]").click()
        time.sleep(2)

        # Log response
        current_url = driver.current_url
        print(f"[➡️] Redirected to: {current_url}")

        if "access-denied" in current_url:
            print("🚫 Access Denied page reached. Stopping further attempts.")
            break

        time.sleep(1)

except Exception as e:
    print("❌ Error occurred:", e)

finally:
    driver.quit()
