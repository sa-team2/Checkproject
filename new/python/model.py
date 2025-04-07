from transformers import pipeline

# 載入 Hugging Face 的 Zero-shot Learning 分類模型
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 保存模型到本地
classifier.model.save_pretrained('./emomodel')
classifier.tokenizer.save_pretrained('./emomodel')

print("模型已成功保存！")


