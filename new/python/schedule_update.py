import os
import sys
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import json
import subprocess
import io

# file_path = sys.argv[1]  # 第一个参数是脚本名，第二个是传入的文件路径

# # 获取当前脚本所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    if len(sys.argv) < 2:
        print("No file path provided.")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    _, ext = os.path.splitext(file_path)

    if ext not in ['.xls', '.xlsx']:
        print(f"Unsupported file type: {ext}")
        return

    try:
        # 讀取 Excel 檔
        df = pd.read_excel(file_path)

        print("Excel file loaded successfully!")

        # 確認欄位存在
        required_columns = ['關鍵字', '是否是詐騙', '類型']
        if not all(col in df.columns for col in required_columns):
            print("Excel 檔案缺少必要欄位：關鍵字、是否是詐騙、類型")
            return

        # 上傳每一筆資料到 Firestore
        added = []
        duplicates = []

        for index, row in df.iterrows():
            keyword = str(row['關鍵字']).strip()
            is_fraud = int(row['是否是詐騙'])
            fraud_type = str(row['類型']).strip()

            # 查詢是否已存在相同 keyword 的資料
            query = db.collection('FraudDefine').where('Keyword', '==', keyword).stream()
            exists = any(query)  # 只要有一筆資料就代表存在

            if exists:
                duplicates.append(keyword)
            else:
                doc = {
                    'Keyword': keyword,
                    'Result': is_fraud,
                    'Type': fraud_type
                }
                db.collection('FraudDefine').add(doc)
                added.append(keyword)
                print(f"✅ 新增關鍵字：{keyword}")

        # 最後輸出結果
        print("\n📄 上傳完成")
        print("🟢 新增的關鍵字：", added if added else "無")
        print("🟡 已存在的關鍵字：", duplicates if duplicates else "無")
        print("✅ 所有資料已成功上傳到 Firestore！")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

    # 刪除檔案
    os.remove(file_path)
    print(f"File {file_path} deleted.")





# 取得目前目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# # 建構三個 script 的路徑
svm_script_path = os.path.join(current_dir, 'svm1.py')
model_script_path = os.path.join(current_dir, 'model.py')

# # 依序執行三個 script
subprocess.run(['python', svm_script_path], check=True)
subprocess.run(['python', model_script_path], check=True)

print("模型更新完畢")



if __name__ == '__main__':
    main()