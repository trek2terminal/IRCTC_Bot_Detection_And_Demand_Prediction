from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Configure Chrome
options = Options()
options.add_argument("--start-maximized")
# options.add_argument("--headless")  # Uncomment to run in headless mode

# Start persistent browser session
driver = webdriver.Chrome(options=options)

try:
    for attempt in range(1, 7):  # 6 login attempts
        print(f"\n🔁 [Attempt #{attempt}]")
        driver.get("http://127.0.0.1:5000")
        time.sleep(1.5)  # Allow page to load

        # Fill credentials
        driver.find_element(By.ID, "username").clear()
        driver.find_element(By.ID, "username").send_keys("bot1")
        driver.find_element(By.ID, "password").clear()
        driver.find_element(By.ID, "password").send_keys("fast123")

        # Read and enter CAPTCHA
        captcha_text = driver.find_element(By.ID, "captchaText").text
        print(f"[🔐] CAPTCHA: {captcha_text}")
        driver.find_element(By.ID, "captchaInput").clear()
        driver.find_element(By.ID, "captchaInput").send_keys(captcha_text)

        # Submit form
        driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]").click()

        # Wait for navigation or error display
        time.sleep(2)
        current_url = driver.current_url
        print(f"[➡️] Redirected to: {current_url}")

        # Stop if blocked
        if "access-denied" in current_url:
            print("🚫 Access Denied page reached. Stopping further attempts.")
            break

        time.sleep(1.5)  # Optional delay before next attempt

except Exception as e:
    print("❌ Error:", e)

finally:
    driver.quit()
