import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入已儲存的訓練好的 BERT 模型
model_path = "bert"  # 用您儲存的模型目錄替換此路徑
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# 載入 Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

def encode_texts(texts, tokenizer, max_length=32):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def predict(text):
    """ 給定文本，回傳詐騙機率 """
    encoded = encode_texts([text], tokenizer)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    fraud_prob = probs[0][1].item() * 100  # 詐騙的機率
    return fraud_prob


if __name__ == "__main__":
    while True:
        text = input("請輸入要測試的文本（輸入 'exit' 退出）: ")
        if text.lower() == 'exit':
            break
        prob = predict(text)
        print(f"【{text}】詐騙可能性: {prob:.2f}%")
