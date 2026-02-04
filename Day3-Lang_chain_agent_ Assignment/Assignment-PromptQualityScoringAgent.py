from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma2:2b", verbose=True)

from langchain_core.prompts import ChatPromptTemplate
system_template = """You are a helpful assistant.

Your task is to evaluate a given prompt and provides scores out of 10 based on the following criteria:
- Clarity: check whether the prompt is easy tp understand and has a clear goal
- Specificity: Evaluate whether sufficient details and requirements are provided
- Context: check if background information, audience or use case is mentioned
- Output format and constraints: check whether expected format, tone or length is specified  
- Persona defined: confirms whether a prompt assigns a specific role

Output: 
- Final Score
- Score for each quality criteria
- short explanation
- 2-3 suggestion to improve the prompt
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", """ what is the weather in Paris today?""")
])

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

promt_quality_chain = prompt_template | llm | parser

def check_prompt_quality (prompt) -> str:
    input = {
        "prompt" : prompt
    }
    prompt_quality = promt_quality_chain.invoke(input)
    return prompt_quality 

# from IPython.display import display, Markdown
# def render_prompt_quality(prompt_quality: str):
#     display(Markdown(prompt_quality))       

result = check_prompt_quality(
    prompt="what is the weather in Paris today?"
)

# render_prompt_quality(result)
print(result)
