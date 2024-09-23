from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR
import firebase_admin
from firebase_admin import credentials, firestore

# 初始化 Flask 應用
app = Flask(__name__)

# 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-cdd57f1038.json')  
firebase_admin.initialize_app(cred)
db = firestore.client()

# 加載模型和處理器
def load_models():
    models = []
    i = 0
    while True:
        try:
            model = joblib.load(f'ocsvm_model_{i}.pkl')
            models.append(model)
            i += 1
        except FileNotFoundError:
            break
    return models

ocsvm_models = load_models()
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)


# 從 Firebase 獲取關鍵字和類型
def get_keywords_from_firebase():
    """從 Firebase 的 FraudDefine 集合中獲取關鍵字和類型"""
    keywords_data = []
    try:
        keywords_ref = db.collection('FraudDefine')
        docs = keywords_ref.stream()
        for doc in docs:
            keyword_info = doc.to_dict()
            if 'Keyword' in keyword_info and 'Type' in keyword_info:
                keywords_data.append({
                    'keyword': keyword_info['Keyword'],  # 獲取 Keyword 
                    'type': keyword_info['Type']  # 獲取 Type 
                })
    except Exception as e:
        print(f"Failed to fetch keywords from Firebase: {e}")
    return keywords_data

# 匹配輸入文本中的關鍵字
def match_keywords(text, keywords_data):
    """在文本中匹配 Firebase 提供的關鍵字"""
    matched = []
    for keyword_data in keywords_data:
        if keyword_data['keyword'] in text:
            matched.append({
                'keyword': keyword_data['keyword'],
                'type': keyword_data['type']
            })
    return matched

# 下載圖片並轉換為 OpenCV 格式
def download_image_from_url(image_url):
    """從 URL 下載圖片並轉為 OpenCV 格式"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        if 'image' not in response.headers.get('Content-Type', ''):
            raise Exception(f"URL does not point to an image: {image_url}")

        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception(f"Failed to decode image: {image_url}")
        return image
    except Exception as e:
        print(f"Failed to download or decode image {image_url}: {e}")
        return None

# 增強圖片
def enhance_image(image, scale_factor=4.0):
    """增強圖片（調整對比度、亮度和銳化）"""
    alpha = 1.5  
    beta = 0    
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# OCR 識別
def perform_ocr(image):
    """執行 OCR 識別"""
    result = ocr.ocr(image, cls=True)
    return result

# 提取 OCR 識別結果中的文字
def extract_text_from_ocr(result):
    """提取 OCR 識別結果中的文字"""
    if not result:
        return ''
    text = ''
    for line in result:
        for word_info in line:
            text += word_info[1][0] + ' '
    return text.strip()

# Sigmoid 函數
def sigmoid(x):
    """Sigmoid 激活函數"""
    return 1 / (1 + np.exp(-x))

# 主預測函數
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')
    image_urls = data.get('image_urls', [])

    # 如果沒有傳遞 image_urls，只處理輸入文本
    if not image_urls:
        combined_text = input_text
    else:
        ocr_texts = []
        ocr_results = {}
        processed_urls_in_request = set()  # 每次請求獨立的URL追蹤集合

        for image_url in image_urls:
            if image_url in processed_urls_in_request:
                print(f"Skipping already processed URL in this request: {image_url}")
                continue
            processed_urls_in_request.add(image_url)  # 只記錄當前請求中的URL

            try:
                image = download_image_from_url(image_url)
                if image is None:
                    continue

                enhanced_image = enhance_image(image)
                ocr_data = perform_ocr(enhanced_image)
                ocr_text = extract_text_from_ocr(ocr_data)
                ocr_texts.append(ocr_text)
                ocr_results[image_url] = ocr_text

            except Exception as e:
                print(f"Failed to process image {image_url}: {e}")
                ocr_results[image_url] = str(e)

        # 將 OCR 文本和輸入文本合併
        combined_text = input_text + ' ' + ' '.join(ocr_texts)

    # 獲取 Firebase 中的關鍵字和類型
    keywords_data = get_keywords_from_firebase()

    # 匹配關鍵字
    matched_keywords = match_keywords(combined_text, keywords_data)

    # 使用模型進行預測
    new_sample_vector = vectorizer.transform([combined_text]).toarray()
    new_sample_scaled = scaler.transform(new_sample_vector)
    new_sample_pca = pca.transform(new_sample_scaled)

    # 獲取每個模型的預測結果和信心值
    predictions = [ocsvm.predict(new_sample_pca)[0] for ocsvm in ocsvm_models]
    scores = [ocsvm.decision_function(new_sample_pca)[0] for ocsvm in ocsvm_models]
    probabilities = sigmoid(np.array(scores))

    # 計算平均信心百分比
    avg_confidence_percentage = np.mean(probabilities) * 100

    result = "非詐騙" if any(pred == 1 for pred in predictions) else "詐騙"

    return jsonify({
        'result': result,
        'matched_keywords': matched_keywords,  # 返回匹配的關鍵字和類型
        'ocr_results': ocr_results if image_urls else {},  # 僅當有 image_urls 時返回 OCR 結果
        'FraudRate': avg_confidence_percentage  # 返回平均信心百分比
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
