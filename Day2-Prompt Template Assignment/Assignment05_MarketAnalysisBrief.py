from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(model = "gemma2:2b")

system_message = """
Role: Yoy're an expert in beifing market anlysis
Task: Generate a market analysis brief from multiple news articles and internal reports provided. 


Rules:
    - include SWOT anlysis, 
    - highlight top 3 trends
    - provide citations

Expected output format:
    - JSON + narrative summary with citation for each trend
 
Guardrails:
    - No frabricated data
    - Ensure all trends are grounded and supported by sources
User question: {question}"""

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template("{question}")
])

username = input("Hey there, May i know your Name please?: ")

while True:
    question = input("Enter your question: ")
    response = llm.invoke(template.format(
                username = username,
                question = question
    ))
    print(f"Assistant : {response.text}")




