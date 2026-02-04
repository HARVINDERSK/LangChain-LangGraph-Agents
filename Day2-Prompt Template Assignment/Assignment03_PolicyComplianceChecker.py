from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(model = "gemma2:2b")

system_message ="""
 Role: You are an HR Policy Compliance expert
 Task: you need to review HR Policy and flag any missing compliance clauses, ambigiuous language and suggest any improvement needed. 
 Rules: 
 - Output JSON with issues, severity and recommendations 
 - cite policy refernces 
 - Do not invent compliance rules
 - suggestion must be based on the provided context
 
 user question: {question}"""

# Output Response format:
#      JSON:{'issues':[],
#          'severity':[],
#          'recomendations':[]}
 


template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template("{question}")    
])
user_name = input("Hey, may i know you name please: ")

question = """Sample HR Policy Document
1. Introduction
This policy outlines the company’s expectations, rights, and responsibilities for all employees. It aims to foster a respectful, safe, and productive workplace.

2. Equal Employment Opportunity (EEO)
We are committed to providing equal employment opportunities to all employees and applicants without regard to race, color, religion, gender, sexual orientation, gender identity, national origin, age, disability, or any other protected status.

3. Code of Conduct
Employees are expected to:

Treat colleagues, clients, and partners with respect.
Maintain honesty and integrity in all business dealings.
Avoid conflicts of interest and report any potential issues to HR.
4. Attendance & Punctuality
Regular attendance is essential. Employees should:

Arrive on time for scheduled shifts.
Notify their supervisor as soon as possible in case of absence or lateness.
Follow leave request procedures.
5. Workplace Safety
We prioritize a safe work environment by:

Complying with all safety regulations.
Reporting hazards or unsafe conditions immediately.
Participating in required safety training.
6. Anti-Harassment & Anti-Discrimination
Harassment of any kind will not be tolerated. Employees should:

Report incidents to HR or management promptly.
Expect all complaints to be investigated confidentially and fairly.
7. Leave & Time Off
We provide:

Paid time off (PTO) for vacation, illness, and personal needs.
Family and medical leave in accordance with applicable laws.
Public holiday observance as per the company calendar.
8. Confidentiality
Employees must protect sensitive company and client information, both during and after employment.

9. Disciplinary Action
Violations of company policy may result in:

Verbal or written warnings.
Suspension.
Termination, depending on severity."""


while True:
    # question = input("Enter your question: ")
    response = llm.invoke(template.format(
                user_name = user_name,
                question = question
    ))

    print(f"Assistant : {response.text}")