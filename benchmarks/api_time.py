import requests
import time
import re

def get_url():
  url = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'
  response = requests.get(url)
  html_text = response.text
  match = re.search(r'<img\s+src=auth_img\.php\?pwdstr=([0-9\-]+)', html_text)
  return url + 'auth_img.php?pwdstr=' + match.group(1)

def get_one():
  global counter
  answer = ''
  url = 'http://ocr.nthumods.com?url=' + get_url()
  response = requests.get(url)
  print(response.text)
  
start_time = time.time()
  
for i in range(10):
  get_one()
  
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")