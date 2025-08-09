from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import string

# --- Password generator ---
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

# --- Username generator ---
def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# --- Setup Chrome driver ---
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 30)

try:
    for attempt in range(1, 11):
        print(f"\n🔁 Attempt #{attempt}")
        driver.get("http://127.0.0.1:5000")
        wait.until(EC.presence_of_element_located((By.ID, "username")))

        username = f"user_{random_string(5)}"
        password = generate_secure_password()

        # Scroll
        driver.execute_script("window.scrollTo({top: 300, behavior: 'smooth'});")
        time.sleep(1)

        # Typing username
        typing_delay = random.uniform(0.1, 0.25)
        username_input = driver.find_element(By.ID, "username")
        username_input.clear()
        for char in username:
            username_input.send_keys(char)
            time.sleep(typing_delay)

        # Typing password
        password_input = driver.find_element(By.ID, "password")
        password_input.clear()
        for char in password:
            password_input.send_keys(char)
            time.sleep(typing_delay)

        # Show a persistent popup inside the page (not alert)
        popup_script = """
        let note = document.createElement('div');
        note.innerText = "⚠️ Please manually enter the CAPTCHA and click the Login button yourself.";
        note.style.position = 'fixed';
        note.style.bottom = '20px';
        note.style.left = '20px';
        note.style.padding = '12px 20px';
        note.style.backgroundColor = '#ffeb3b';
        note.style.color = '#000';
        note.style.fontSize = '16px';
        note.style.fontWeight = 'bold';
        note.style.border = '2px solid #000';
        note.style.borderRadius = '6px';
        note.style.boxShadow = '0 0 10px rgba(0,0,0,0.3)';
        note.style.zIndex = 9999;
        document.body.appendChild(note);
        """
        driver.execute_script(popup_script)

        print("🧠 CAPTCHA alert shown on page. Waiting for user to manually submit...")

        # Now wait until page is redirected or user manually logs in
        timeout = 60  # seconds to wait for user
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_url = driver.current_url
            if "welcome" in current_url or "access-denied" in current_url:
                print(f"[➡️] Redirected to: {current_url}")
                if "access-denied" in current_url:
                    print("🚫 Access Denied page reached. Stopping further attempts.")
                    break
                break
            time.sleep(1)
        else:
            print("⏰ Timeout: User didn’t submit login in expected time.")

        time.sleep(2)

except Exception as e:
    print("❌ Error:", e)

finally:
    driver.quit()
