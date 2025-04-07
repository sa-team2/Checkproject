from flask import Flask, request, jsonify
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# 初始化 Flask 應用
app = Flask(__name__)

# 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json')
firebase_admin.initialize_app(cred , name='app')
fapp = firebase_admin.get_app('app')  # 確保使用已初始化的 'app' 實例
db = firestore.client(fapp)  # 使用這個實例連接 Firestore

from check_text import check_text_for_lineid_and_url





# 主預測函数
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')
    image_urls = data.get('image_urls', [])
    ocr_texts=[]
    # 如果没有傳遞 image_urls，只處理輸入文本
    if  image_urls==[]:
        combined_text = input_text
    else:
        from ocr import process_images
        ocr_texts,ocr_results=process_images(image_urls)

    combined_text = input_text + ' ' + ' '.join(ocr_texts)

    # 將 OCR 文本和輸入文本合並
    
    text, lineid_data, url_data = check_text_for_lineid_and_url(combined_text)

    if not lineid_data and not url_data:    

        from keywords import get_and_match_keywords_with_details
        from fraud_model import predict
        from roletest import interactive_input

        matched_keywords = get_and_match_keywords_with_details(text)
        fraud_probability = predict(text)
        max_emotion,max_score=interactive_input(text)
        # 按照比例缩放欺诈可能性
        scaled_fraud_probability = fraud_probability * 0.9  # 90% 是欺诈概率
        scaled_max_score = max_score * 100 * 0.1  # 10% 是 max_score * 100

        # 计算总的输出（百分比）
        total_probability = scaled_fraud_probability + scaled_max_score

        # 输出结果
        print(f"詐騙可能性: {fraud_probability:.2f}%")
        print(f"\n🌟 累計情緒分數：最高情緒是 '{max_emotion}'，總分：{max_score:.2f}")
        print(f"⏳ 最終詐騙可能性（基於縮放比例）：{total_probability:.2f}%")

        return jsonify({
            'result': '詐騙' if total_probability >= 50 else '非詐騙',  # 超过50%即为詐騙
            'matched_keywords': matched_keywords, # 返回匹配的关键字和类型
            'ocr_results': ocr_results if image_urls else {},  # 仅当有 image_urls 时返回 OCR 结果
            'FraudRate': total_probability , # 返回平均置信度百分比
            'Emotion':max_emotion
        })
    else:
        # 构建 matched_keywords 列表，包含匹配的关键字和类型
        matched_keywords = []
        if lineid_data:
            for lineid_data in lineid_data:
                matched_keywords.append({
                    'keyword': lineid_data.get('LineID', '无 LineID'),
                    'type': lineid_data.get('Type', '无类型'),
                    'Remind': lineid_data.get('GoverURL', '无 GoverURL'),
                    'Prevent': lineid_data.get('Prevent', '无 Prevent')
                })
        
        if url_data:
            # 假设 url_info 是从 check_text_for_lineid_and_url 函数中得到的
            for url_data in url_data:
                matched_keywords.append({
                    'keyword': url_data.get('url', '无 URL'),
                    'type': url_data.get('Type', '无类型'),
                    'Remind': url_data.get('GoverURL', '无 GoverURL'),
                    'Prevent': url_data.get('Prevent', '无 Prevent')
                })

        # 构造 JSON 响应
        return jsonify({
            'result': '詐騙',  # 超过50%即为詐騙
            'matched_keywords': matched_keywords,  # 返回匹配的关键字和类型
            'ocr_results': ocr_results if image_urls else {},  # 仅当有 image_urls 时返回 OCR 结果
            'FraudRate': 100  # 返回平均置信度百分比
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)