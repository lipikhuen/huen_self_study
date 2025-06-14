# -*- coding: utf-8 -*-
"""
动物图像识别作业：自动识别图片中的动物，并与人工标注标签比对，输出准确率与详细结果。

作者：李碧萱 2023152605
本脚本和 images 文件夹、result.csv 应放在同一目录下。
images 文件夹中需包含 labels.txt 和所有待识别的动物图片。
"""

import requests
import base64
import urllib.parse
import os
import time
import csv

# ====== 1. 配置区 ======
API_KEY = "7MiVjI3LXl6vtkSBQ1Qw6AuS"
SECRET_KEY = "2SP8RLiXlv2MBiuyf6YfGpDXrgMA61Kj"

# 获取本脚本所在目录（确保无论从哪里启动都正确）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_FILE = os.path.join(IMAGES_DIR, "labels.txt")
RESULT_FILE = os.path.join(BASE_DIR, "result.csv")

# ====== 2. 获取Access Token ======
def get_access_token(api_key, secret_key):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    res = requests.post(url, params=params)
    res_json = res.json()
    if "access_token" in res_json:
        return res_json["access_token"]
    else:
        raise Exception(f"获取access_token失败：{res_json}")

# ====== 3. 加载人工标签 ======
def load_labels(labels_file):
    labels = {}
    with open(labels_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            fname, label = line.split(',', 1)
            labels[fname.strip()] = label.strip()
    return labels

# ====== 4. 调用API进行识别 ======
def recognize_animal(image_path, access_token):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    base64_data = base64.b64encode(img_data).decode()
    base64_data = urllib.parse.quote_plus(base64_data)
    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v1/animal?access_token={access_token}"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = f"image={base64_data}&top_num=1"
    try:
        res = requests.post(url, data=data, headers=headers, timeout=10)
        return res.json()
    except Exception as e:
        print(f"请求出错：{e}")
        return {}

# ====== 5. 主流程 ======
def main():
    print("1. 获取access_token ...")
    access_token = get_access_token(API_KEY, SECRET_KEY)
    print("2. 加载labels.txt ...")
    labels = load_labels(LABELS_FILE)
    print(f"3. 检测到 {len(labels)} 张图片需要识别。")

    # 支持图片格式
    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith(('.jpg','.jpeg','.png','.bmp')) and f in labels]

    results = []
    correct_count = 0

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        print(f"正在识别: {img_name} ...")
        api_result = recognize_animal(img_path, access_token)
        if "result" in api_result and api_result["result"]:
            top_result = api_result["result"][0]
            pred_name = top_result.get("name", "")
            score = top_result.get("score", "")
        else:
            pred_name = "无法识别"
            score = "0"

        true_label = labels.get(img_name, "未知")
        is_correct = int(pred_name == true_label)
        if is_correct:
            correct_count += 1
        results.append([img_name, pred_name, score, true_label, "正确" if is_correct else "错误"])
        time.sleep(0.4)  # 防止被限流

    # ====== 6. 保存结果到 result.csv ======
    with open(RESULT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["图片名", "API识别", "置信度", "人工标签", "识别是否正确"])
        for row in results:
            writer.writerow(row)

    accuracy = correct_count / len(results) if results else 0
    print(f"\n全部识别完成！准确率：{accuracy:.2%}")
    print(f"识别结果已保存到 {RESULT_FILE}")

if __name__ == "__main__":
    main()
