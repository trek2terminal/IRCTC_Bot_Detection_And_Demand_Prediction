# bot3_random_scroll.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time, random

options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

try:
    driver.get("http://127.0.0.1:5000")
    time.sleep(1)

    # Simulate scroll to fake human
    driver.execute_script(f"window.scrollTo(0, {random.randint(50, 300)});")

    driver.find_element(By.ID, "username").send_keys("bot3")
    time.sleep(0.5)
    driver.find_element(By.ID, "password").send_keys("scroll123")

    captcha = driver.find_element(By.ID, "captchaText").text
    driver.find_element(By.ID, "captchaInput").send_keys(captcha)

    driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]").click()
    time.sleep(2)
    print("[✅ Bot3] Redirected to:", driver.current_url)

finally:
    driver.quit()
