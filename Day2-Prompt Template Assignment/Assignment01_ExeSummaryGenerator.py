from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(model = "gemma2:2b")

system_message ="""
 You need to generate Executive Summary
 Rules: 
 - Summarize a 10-page quterly performance report into an executive summary with key metrics and risks"
 - prompt needs to be role based
 - include metrics table and risk section
 - fact based only
 - no hallucination
 - cite source if required 
 Response format:
 - 150 word summary
 - includes table of metrics
 - show riks and/or opportunites section
 user question: {question}"""

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template("{question}")    
])
user_name = input("Hey, may i know you name please: ")

while True:
    question = input("Enter your question")
    response = llm.invoke(template.format(
                user_name = user_name,
                question = question
    ))

    print(f"Assistant : {response.text}")