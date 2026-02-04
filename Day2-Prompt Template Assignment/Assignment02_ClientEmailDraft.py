from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(model = "gemma2:2b")

system_message = """
you need to draft a professional emal to client summarizing project progress, milestone, and requesting feedback.
Rules:
    - keep a formal tone
    - keep a placeholder for client name, project name, deadline, and include action items

Expected output format:
    - Email text with a subject line 
    - Bullet point for the action items

Guardrails:
    - you neeed to maintina a professional tone
    - avoid any sensitive data
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




