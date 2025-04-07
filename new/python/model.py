from transformers import pipeline

# 載入 Hugging Face 的 Zero-shot Learning 分類模型
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 保存模型到本地
classifier.model.save_pretrained('./model')
classifier.tokenizer.save_pretrained('./model')

print("模型已成功保存！")


from transformers import BertTokenizer, BertForSequenceClassification
import os

def download_model_and_save():
    # 设置本地保存的文件夹路径
    model_dir = "emo_model"
    os.makedirs(model_dir, exist_ok=True)  # 如果文件夹不存在，则创建文件夹

    # 下载和保存 BERT 模型和分词器
    print("正在下载模型和分词器...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
    
    # 保存到本地
    print(f"正在将模型和分词器保存到 {model_dir} 文件夹...")
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    
    print(f"模型和分词器已成功保存到 {model_dir} 文件夹。")

if __name__ == "__main__":
    download_model_and_save()

