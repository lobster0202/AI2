# STEP 1 - 5 40 
from transformers import pipeline

# AutoTokenizer는 파이프라인이 없을 때 사용할 수 있다. 
# from transformers import AutoTokenizer

# from transformers import AutoModelForSequenceClassification



# STEP 2
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")

# model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")


# STEP 3
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP 4
result = classifier(text)
# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# STEP 5
print(result)






