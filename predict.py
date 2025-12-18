import requests
import os
from tqdm import tqdm
import json
import csv
import time
from datetime import datetime


# =========================================================
# Load API Keys
# =========================================================
with open('api-keys.json', 'r', encoding='utf-8') as f:
    api_keys = json.load(f)

llm_small = api_keys[2]
AUTHORIZATION = llm_small["authorization"]
TOKEN_KEY = llm_small["tokenKey"]
TOKEN_ID = llm_small["tokenId"]

API_URL = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small"

# =========================================================
# Config quota
# =========================================================
MAX_REQ_PER_HOUR = 60
MAX_REQ_PER_DAY = 1000

# =========================================================
# Load dataset
# =========================================================
data_path = "./src/data/test.json"
output_path = "submission.csv"

with open(data_path, 'r', encoding="utf-8") as f:
    dataset = json.load(f)


# =========================================================
# Hàm gọi API với retry
# =========================================================
def call_api_with_retry(payload, max_retries=5, backoff=2):
    headers = {
        "Authorization": AUTHORIZATION,
        "Token-id": TOKEN_ID,
        "Token-key": TOKEN_KEY,
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)

            if response.status_code == 200:
                return response.json()

            # Lỗi quota hoặc server → retry
            if response.status_code in [429, 500, 502, 503, 504]:
                wait = backoff ** attempt
                print(f"⚠️ API error {response.status_code}, retry sau {wait}s...")
                time.sleep(wait)
                continue

            print(f"❌ API error {response.status_code}: {response.text}")
            return None

        except Exception as e:
            wait = backoff ** attempt
            print(f"⚠️ Exception: {e}, retry sau {wait}s...")
            time.sleep(wait)

    print("❌ Hết số lần retry")
    return None


# =========================================================
# Hàm predict
# =========================================================
def predict(question, choices):
    payload = {
        'model': "vnptai_hackathon_small",
        'messages': [
            {
                'role': 'system',
                'content': """
                    Bạn là hệ thống trả lời trắc nghiệm. Nhiệm vụ:
                    - Mỗi câu hỏi có danh sách lựa chọn. CHỈ trả về ĐÚNG MỘT KÝ TỰ mã đáp án (A, B, C, D, …). Không giải thích, không thêm dấu câu.
                    - Nếu đề bài kèm đoạn thông tin, phải ưu tiên suy luận từ đoạn đó; chỉ dùng kiến thức chung khi đoạn không đủ thông tin.
                    - Với bài toán có ký hiệu toán, xử lý chính xác công thức.
                    - Nếu không chắc chắn, chọn đáp án khả dĩ nhất, không sinh “không biết”.
                    Định dạng trả lời: chỉ một ký tự đáp án (vd: B)
                """
            },
            {
                'role': 'user',
                'content': f"Hãy trả lời câu hỏi sau:\n{question}\nĐáp án:\n{choices}",
            }
        ],
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 20,
        'n': 1,
        'max_completion_tokens': 1,
    }

    result = call_api_with_retry(payload)

    if result and "choices" in result:
        return result["choices"][0]["message"]["content"].strip()

    return "1"  # fallback


# =========================================================
# Kiểm tra file submission → load progress cũ nếu có
# =========================================================
def load_progress():
    if not os.path.exists(output_path):
        return 0

    with open(output_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))

    if len(reader) <= 1:
        return 0

    return len(reader) - 1  # bỏ dòng header


# =========================================================
# MAIN – chạy theo batch, kiểm soát quota
# =========================================================
start_index = load_progress()
total = len(dataset)
print(f"👉 Bắt đầu từ index {start_index}/{total}")

# Nếu file chưa tồn tại → tạo header
if start_index == 0:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer"])

hour_counter = 0
day_counter = start_index  # đã dùng bao nhiêu request hôm nay

current_hour = datetime.now().hour


# =========================================================
# Vòng lặp chính
# =========================================================
with open(output_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    for idx in tqdm(range(start_index, total)):

        # Reset quota theo giờ
        now_hour = datetime.now().hour
        if now_hour != current_hour:
            current_hour = now_hour
            hour_counter = 0
            print("🔄 Reset quota giờ")

        # Nếu vượt quota theo giờ → chờ
        if hour_counter >= MAX_REQ_PER_HOUR:
            print("⏳ Đã dùng 60 req/h, chờ 1 giờ...")
            time.sleep(3600)
            hour_counter = 0

        # Nếu vượt quota ngày → dừng script
        if day_counter >= MAX_REQ_PER_DAY:
            print("❌ Đã dùng hết 1000 req/ngày → dừng lại.")
            break

        item = dataset[idx]
        qid = item['qid']
        question = item['question']
        choices = "\n".join(item['choices'])

        answer = predict(question, choices)

        writer.writerow([qid, answer])
        f.flush()

        hour_counter += 1
        day_counter += 1

        time.sleep(0.5)  # giảm tốc độ để tránh spam API

print("🎉 Hoàn thành batch chạy.")
