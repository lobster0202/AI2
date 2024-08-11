# STEP 1 - 임포트
from transformers import pipeline

# STEP 2 - 파이프 라인에서 summarization 불러오기
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

# STEP 3 - 모델에 넣을 text 선언하기 
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

# STEP 4 - summarizer에 text 넣고 결과 값 result에 선언하기
result = summarizer(text)

# STEP 5 - 결과 값인 result 출력하기
print(result)