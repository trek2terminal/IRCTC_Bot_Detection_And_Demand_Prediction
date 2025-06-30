# bot1_fast_typing.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

try:
    driver.get("http://127.0.0.1:5000")
    time.sleep(1)

    driver.find_element(By.ID, "username").send_keys("bot1")
    driver.find_element(By.ID, "password").send_keys("fast123")

    captcha = driver.find_element(By.ID, "captchaText").text
    driver.find_element(By.ID, "captchaInput").send_keys(captcha)

    driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]").click()
    time.sleep(3)
    print("[✅ Bot1] Redirected to:", driver.current_url)

except Exception as e:
    print("[❌ Bot1]", e)
finally:
    driver.quit()
