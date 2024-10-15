from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR
import firebase_admin
from firebase_admin import credentials, firestore
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# 初始化 Flask 應用
app = Flask(__name__)

# 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-23ed2646dd.json')  # 替换为你的 Firebase 服务密钥路径
firebase_admin.initialize_app(cred)
db = firestore.client()

# 初始化 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)



# 加載判斷是否是詐騙的模型1和處理器
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


# 加载詐騙類型的模型2和嵌入
checkpoint = torch.load('bert_fraud_model.pth')
model_type = BertModel.from_pretrained('bert-base-chinese')
model_type.load_state_dict(checkpoint['model_state_dict'])
model_type.eval()
fraud_type_embeddings = checkpoint['fraud_type_embeddings']



# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

# 用于跟蹤處理過的圖片 URL
processed_urls = set()

DEFAULT_TYPE = "未知"  # 默认类型

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
                    'keyword': keyword_info['Keyword'],
                    'type': keyword_info['Type']
                })
    except Exception as e:
        print(f"Failed to fetch keywords from Firebase: {e}")
    return keywords_data

def match_keywords(text, keywords_data):
    """匹配或预测文本中的关键字及其类型"""
    matched = [
        {'keyword': kd['keyword'], 'type': kd['type']}
        for kd in keywords_data if kd['keyword'] in text
    ]

    if not matched:  # 如果没有匹配到关键字
        suspicious_keywords = find_suspicious_keywords(text, keywords_data)
        predicted_types = predict_fraud_types([kw for kw, _ in suspicious_keywords])
        
        # 如果预测失败，返回默认类型
        matched.extend({
            'keyword': kw, 
            'type': pt if pt else DEFAULT_TYPE
        } for kw, pt in predicted_types)

    return matched

def predict_fraud_types(keywords, top_n=1):
    """使用 BERT 模型预测关键词的类型"""
    global fraud_type_embeddings  # 已有的诈骗类型嵌入

    predicted_types = []
    for keyword in keywords:
        keyword_embedding = get_bert_embedding(keyword)

        # 计算与所有诈骗类型的相似度
        similarities = {
            fraud_type: np.dot(embedding.flatten(), keyword_embedding.flatten())
            for fraud_type, embedding in fraud_type_embeddings.items()
        }

        # 按相似度排序，选出最相似的类型
        sorted_types = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        if sorted_types and sorted_types[0][1] > 0.5:  # 阈值为 0.5
            predicted_types.append((keyword, sorted_types[0][0]))
        else:
            predicted_types.append((keyword, DEFAULT_TYPE))

    return predicted_types

def get_bert_embedding(text): 
    """通过 BERT 获取文本的嵌入向量"""
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model_type(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def find_suspicious_keywords(text, keywords_data):
    """使用 TF-IDF 提取最可疑关键词"""
    
    # 这里我们不拆分为单个词，而是直接将文本传给 TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])  # 直接使用原始文本
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # 选取 TF-IDF 分数最高的两个关键词
    top_indices = tfidf_scores.argsort()[-2:][::-1]
    suspicious_keywords = [feature_names[i] for i in top_indices]

    # 检查关键词是否已在数据库中
    classified_keywords = [
        (kw, kd['type']) for kw in suspicious_keywords
        for kd in keywords_data if kw == kd['keyword']
    ]

    # 若不在数据库中，进行预测
    if not classified_keywords:
        predicted_types = predict_fraud_types(suspicious_keywords)
        classified_keywords = [(kw, pt if pt else DEFAULT_TYPE) for kw, pt in predicted_types]

    return classified_keywords




# 下載圖片並轉換為 OpenCV 格式
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
    

def load_image_from_path(image_url):
    """从本地路径加载图像并转为 OpenCV 格式"""
    try:
        image = cv2.imdecode(np.fromfile(image_url, dtype=np.uint8), -1)
        if image is None:
            raise Exception(f"Failed to load image from path: {image_url}")
        return image
    except Exception as e:
        print(f"Error loading image from local path: {e}")
        raise  # 抛出异常，便于更好地调试



# 增強圖片
def enhance_image(image, scale_factor=4.0):
    """增強圖片（調整對比度、亮度和銳化）"""
    alpha = 1.5  # 對比度
    beta = 0     # 亮度
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# OCR 識别
def perform_ocr(image):
    """执行 OCR 识别"""
    result = ocr.ocr(image, cls=True)
    return result

# 提取 OCR 識别結果中的文字
def extract_text_from_ocr(result):
    """提取 OCR 識别結果中的文字"""
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

def is_url(path):
    """判断传递的路径是否为 URL"""
    return path.startswith(('http://', 'https://'))


# 主預測函数
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')
    image_urls = data.get('image_urls', [])

    # 如果没有傳遞 image_urls，只處理輸入文本
    if  image_urls==[]:
        combined_text = input_text
    else:
        ocr_texts = []
        ocr_results = {}

        for image_url in image_urls:
            if (image_url in processed_urls) and is_url(image_url):
                print(f"Skipping already processed URL: {image_url}")
                continue
            processed_urls.add(image_url)

            try:
                if is_url(image_url):
                # 处理 URL
                    image = download_image_from_url(image_url)
                else:
                    image = load_image_from_path(image_urls)  # 加载本地文件

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

            if (is_url(image_url)==False):
                break

        combined_text = input_text + ' ' + ' '.join(ocr_texts)

    # 將 OCR 文本和輸入文本合並
    

    # 获取 Firebase 中的关键字和类型
    from transformers import BertTokenizer, BertModel
    import torch
    from sklearn.decomposition import PCA
    def encode_with_bert(data):
    
        inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():  # 不需要反向传播
            outputs = model_type(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # 平均池化
        return embeddings.cpu().numpy()

    keywords_data = get_keywords_from_firebase()

    # 匹配关键字
    matched_keywords = match_keywords(combined_text, keywords_data)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_type = BertModel.from_pretrained('bert-base-uncased')

    X_bert_new = encode_with_bert([combined_text])
    # 使用模型进行预测
    new_sample_vector = vectorizer.transform([combined_text]).toarray()

    combined_features = np.concatenate((new_sample_vector, X_bert_new), axis=1)
    new_sample_scaled = scaler.transform(combined_features)
    new_sample_pca = pca.transform(new_sample_scaled)

    # 获取每个模型的预测结果和置信度
    predictions = [ocsvm.predict(new_sample_pca)[0] for ocsvm in ocsvm_models]
    scores = [ocsvm.decision_function(new_sample_pca)[0] for ocsvm in ocsvm_models]
    probabilities = sigmoid(np.array(scores))

    # 设置置信度阈值
    # confidence_threshold = 0.6  # 假设我们使用60%置信度作为阈值

    fraud_probability = 1 - np.mean(probabilities)  # 反转信任度为诈骗概率

    # 计算平均置信度百分比
    avg_fraud_percentage = fraud_probability * 100

    # # 判断是否达到阈值
    # if any(prob >= confidence_threshold for prob in probabilities):
    #     result = "非詐騙"  # 如果至少有一个模型的置信度达到阈值，则判断为非诈骗
    # else:
    #     result = "詐騙"  # 如果所有模型的置信度都低于阈值，则判断为诈骗

    return jsonify({
        'result': '詐騙' if avg_fraud_percentage >= 50 else '非詐騙',  # 超过50%即为詐騙
        'matched_keywords': matched_keywords, # 返回匹配的关键字和类型
        'ocr_results': ocr_results if image_urls else {},  # 仅当有 image_urls 时返回 OCR 结果
        'FraudRate': avg_fraud_percentage  # 返回平均置信度百分比
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
