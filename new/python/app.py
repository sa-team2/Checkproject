from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR
import firebase_admin
from firebase_admin import credentials, firestore

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}) # 允许访问

# 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-cdd57f1038.json')  # 替换为你的 Firebase 服务密钥路径
firebase_admin.initialize_app(cred)
db = firestore.client()

# 加载模型和处理器
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

# 用于跟踪处理过的图片 URL
processed_urls = set()

# 从 Firebase 获取关键字和类型
def get_keywords_from_firebase():
    """从 Firebase 的 FraudDefine 集合中获取关键字和类型"""
    keywords_data = []
    try:
        keywords_ref = db.collection('FraudDefine')
        docs = keywords_ref.stream()
        for doc in docs:
            keyword_info = doc.to_dict()
            if 'Keyword' in keyword_info and 'Type' in keyword_info:
                keywords_data.append({
                    'keyword': keyword_info['Keyword'],  # 获取 Keyword 字段
                    'type': keyword_info['Type']  # 获取 Type 字段
                })
    except Exception as e:
        print(f"Failed to fetch keywords from Firebase: {e}")
    return keywords_data

# 匹配输入文本中的关键字
def match_keywords(text, keywords_data):
    """在文本中匹配 Firebase 提供的关键字"""
    matched = []
    for keyword_data in keywords_data:
        if keyword_data['keyword'] in text:
            matched.append({
                'keyword': keyword_data['keyword'],
                'type': keyword_data['type']
            })
    return matched

# 下载图片并转换为 OpenCV 格式
def download_image_from_url(image_url):
    """从 URL 下载图片并转为 OpenCV 格式"""
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

# 增强图片
def enhance_image(image, scale_factor=4.0):
    """增强图片（调整对比度、亮度和锐化）"""
    alpha = 1.5  # 对比度
    beta = 0     # 亮度
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# OCR 识别
def perform_ocr(image):
    """执行 OCR 识别"""
    result = ocr.ocr(image, cls=True)
    return result

# 提取 OCR 识别结果中的文字
def extract_text_from_ocr(result):
    """提取 OCR 识别结果中的文字"""
    if not result:
        return ''
    text = ''
    for line in result:
        for word_info in line:
            text += word_info[1][0] + ' '
    return text.strip()

# Sigmoid 函数
def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

# 主预测函数
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')
    image_urls = data.get('image_urls', [])

    # 如果没有传递 image_urls，只处理输入文本
    if not image_urls:
        combined_text = input_text
    else:
        ocr_texts = []
        ocr_results = {}

        for image_url in image_urls:
            if image_url in processed_urls:
                print(f"Skipping already processed URL: {image_url}")
                continue
            processed_urls.add(image_url)

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

        # 将 OCR 文本和输入文本合并
        combined_text = input_text + ' ' + ' '.join(ocr_texts)

    # 获取 Firebase 中的关键字和类型
    keywords_data = get_keywords_from_firebase()

    # 匹配关键字
    matched_keywords = match_keywords(combined_text, keywords_data)

    # 使用模型进行预测
    new_sample_vector = vectorizer.transform([combined_text]).toarray()
    new_sample_scaled = scaler.transform(new_sample_vector)
    new_sample_pca = pca.transform(new_sample_scaled)

    # 获取每个模型的预测结果和置信度
    predictions = [ocsvm.predict(new_sample_pca)[0] for ocsvm in ocsvm_models]
    scores = [ocsvm.decision_function(new_sample_pca)[0] for ocsvm in ocsvm_models]
    probabilities = sigmoid(np.array(scores))

    # 计算平均置信度百分比
    avg_confidence_percentage = np.mean(probabilities) * 100

    result = "非詐騙" if any(pred == 1 for pred in predictions) else "詐騙"

    return jsonify({
        'result': result,
        'matched_keywords': matched_keywords,  # 返回匹配的关键字和类型
        'ocr_results': ocr_results if image_urls else {},  # 仅当有 image_urls 时返回 OCR 结果
        'FraudRate': avg_confidence_percentage  # 返回平均置信度百分比
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
