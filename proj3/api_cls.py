# STEP 1
from transformers import pipeline
from fastapi import FastAPI, Form

# STEP 2
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
question_answerer = pipeline("question-answering", model="my_awesome_qa_model")

app = FastAPI()

@app.post("/summarization/")
async def summarization(text: str = Form()) :
    result = summarizer(text)
    return {"result" : result}

@app.post("/classification/")
async def summarization(text: str = Form()) :
    result = classifier(text)
    return {"result" : result}

@app.post("/question_answerer/")
async def question_answerer(question: str = Form(), ) :
    result = summarizer(text)
    return {"result" : result}

# ~~~ 요청받기
# ~~~ 실행  데코레이터 사용해서 여기만 바꿔치기 가능하게 만듬
# ~~~ 반환