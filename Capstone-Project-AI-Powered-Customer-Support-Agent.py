from langchain_ollama import ChatOllama
from typing_extensions import TypedDict, Literal
from typing import Optional, List, Dict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from datetime import datetime
import json

#-----------------------------------------------
# Define the structure for email classification
#-----------------------------------------------
class EmailClassification(TypedDict):
    intent: Literal["question","bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    #Raw email data
    email_content: str
    sender_email: str
    email_id: str

    #Classification result
    classification: Optional[EmailClassification]

    #Raw search/API Result
    search_result: Optional[List[str]]
    customer_history: Optional[Dict]

    #Generated Content
    draft_response: Optional[str]
    messages: Optional[List[str]]

#-------------------------------------
# Initialize the LLM once
#-------------------------------------
llm = ChatOllama(model = "qwen3:0.6b")

#-------------------------------------
# Node: read email
#-------------------------------------
def read_email(state: EmailAgentState)->EmailAgentState:
    """
    Node to read the email content and save in the email_content, sender_email and email_id.    
    """
    # state["email_content"] = "How do I reset my password"
    # state["email_content"] = "The export feature crashes when I select PDF format"
    # state["email_content"] = "I was charged twice for my subscription"
    # state["email_content"] = "Can you add dark mode to the mobile app?"
    state["email_content"] = "Our API integration fails intermittently with a 504 error"
    state["sender_email"] = "Harry" 
    state["email_id"] = "harry@test.com"
    return state

def classify_intent(state: EmailAgentState) -> EmailAgentState:
    """
    Here we will call LLM to classify the intent and urgency of the email.
    """
    prompt = f"""You are a helpful customer assistant. Your task is to classify the customer email.
    
Email content: {state['email_content']}

Classify the email with:
- Intent: one of ["question", "bug", "billing", "feature", "complex"]
- Urgency: one of ["low", "medium", "high", "critical"]
- Topic: a brief topic description
- Summary: a brief summary of the email

Respond in JSON format with keys: intent, urgency, topic, summary"""
    
    response = llm.invoke(prompt)
    data =  json.loads(response.content)

    # Parse the response (simplified - in production you'd use structured output)
    # For now, create a basic classification
    classification: EmailClassification = {
        # "intent": response.content.json()['intent'],  # Default - in production parse from LLM response
        # "urgency": response.content.json()['urgency'],   # Default - in production parse from LLM response
        # "topic": response.content.json()['topic'],  # Default - in production parse from LLM response
        # "summary": response.content[:200] if response.content else "Email about password reset"        
        "intent": data["intent"],
        "urgency": data["urgency"],
        "topic": data["topic"],
        "summary": data["summary"],
    }    
    state["classification"] = classification
    return state
    
#-------------------------------------
# Node: Document Search
#-------------------------------------
def doc_search(state: EmailAgentState) -> EmailAgentState:  
    """
    Node to search the document in the vector store.
    """
    state["messages"] = ["Calling Doc Search -> Searching the document in the vector store for the customer issue"]    
    state["search_result"] = ["Document 1", "Document 2", "Document 3"]
    print("Message :", state['messages'])
    # print("Message :", state['search_result'])

    return state

#-------------------------------------
# Node: Bug Tracking
#-------------------------------------
def bug_track(state: EmailAgentState) -> EmailAgentState:
    """Create or update the bug in the bug tracking system."""
    state["messages"] = ["Invoking Bug Tracking- Creating or updating the bug in the bug tracking system"]
    print("Message :", state['messages'])
    return state

#-------------------------------------
# Node: Human Review
#-------------------------------------
def human_review(state: EmailAgentState) -> EmailAgentState:
    """Escalate tp human agent for appoval or handling"""
    state["messages"] = "Sending for Human Review->Escalating to human agent for appoval or handling"
    print("Message :", state['messages'])
    return state
#-------------------------------------
# Node: Generate draft response email
#-------------------------------------
def draft_reply(state: EmailAgentState)-> EmailAgentState:
    """
    Based on the leave email review, write an email back to the user with an appropriate response
    """
    # Include email content and classification in the prompt
    email_content = state.get('email_content', '')
    customer_name = state.get('sender_email', 'Customer')
    customer_email = state.get('email_id', '')
    classification_info = ""
    if state.get('classification'):
        classification_info = f"\nIssue Type: {state['classification']['intent']}\nUrgency: {state['classification']['urgency']}\nTopic: {state['classification']['topic']}"
    
    prompt = f"""Write a short, professional, and helpful email response to the customer and also schedule follow ups if needed. 

Customer Name: {customer_name}
Customer Email: {customer_email}
Customer's Issue: {email_content}
{classification_info}

Write a clear and concise response addressing their issue.""" 
    draft_response = llm.invoke(prompt)
    state["draft_response"] = draft_response.content
    return state

#-------------------------------------
# Node: Send out email reply
#-------------------------------------
def send_reply(state: EmailAgentState)-> EmailAgentState:
    state["messages"] = "Sending auto-reply email"
    print("Message :", state['messages'])
    return state
#-------------------------------------
# Node: method to help with next node
#-------------------------------------
def decide_next_based_on_intent(state: EmailAgentState) -> str:
    return state["classification"]["intent"]

#-------------------------------------
# Node: method to help with next node
#-------------------------------------
def decide_next_based_on_urgency(state: EmailAgentState) -> str:
    return state["classification"]["urgency"]

#-------------------------------------
# Node: Build the LangGraph
#-------------------------------------
graph = StateGraph(EmailAgentState)

# Register Nodes
graph.add_node("read_email_node", read_email)
graph.add_node("classify_intent_node", classify_intent)
graph.add_node("doc_search_node", doc_search)
graph.add_node("bug_track_node", bug_track)
graph.add_node("human_review_node", human_review)
graph.add_node("draft_reply_node", draft_reply)
graph.add_node("human_review2_node", human_review)
graph.add_node("send_reply_node", send_reply)

#Define Execution flow:
graph.add_edge(START, "read_email_node")
graph.add_edge("read_email_node", "classify_intent_node")

graph.add_conditional_edges(
    "classify_intent_node",
    decide_next_based_on_intent,
    {
        "bug": "bug_track_node",
        "complex": "human_review_node",
        "feature": "doc_search_node",
        "billing": "doc_search_node",
        "question": "doc_search_node",
    },
)

# graph.add_conditional_edges("classify_intent_node", classify_intent, {classification['intent']: "doc_search_node" for classification in ['EmailClassification'] })
# graph.add_conditional_edges("classify_intent_node", classify_intent, lambda state: state.get("classification", {}).get("intent") if state.get("classification") else None, {"bug": "bug_track_node", "complex": "human_review_node", "doc_search": "doc_search_node"}) 
# graph.add_conditional_edges("classify_intent_node", EmailAgentState['classification']['intent'], {"bug": "bug_track_node", "complex": "human_review_node", "doc_search": "doc_search_node"})
# graph.add_edge("classify_intent_node", "bug_track_node", "complex": "human_review_node", "doc_search": "doc_search_node", "default": "draft_reply_node")

graph.add_edge("doc_search_node", "draft_reply_node")
graph.add_edge("bug_track_node", "draft_reply_node")
graph.add_edge("human_review_node", "draft_reply_node")

graph.add_conditional_edges(
    "draft_reply_node",
    decide_next_based_on_urgency,
    {
        "critical": "human_review2_node",
        "high": "human_review2_node",
        "medium": "send_reply_node",
        "low": "send_reply_node",
    },
)

graph.add_edge("human_review2_node", END)
graph.add_edge("send_reply_node", END)

#compile the graph
graph = graph.compile()

#-------------------------------------
# Node: Run the demo
#-------------------------------------
initial_state = {
    "email_content": "",
    "sender_email": "",
    "email_id": "",
    "topic": "",
    "classification": None,
    "search_result": None,
    "customer_history": None,
    "draft_response": None,
    "messages": None,
}
result = graph.invoke(initial_state)

# Access classification fields correctly (they're nested in classification)
if result.get("classification"):
    print("\n" + "="*60)
    print("CLASSIFICATION:")
    print("="*60)
    print("Intent:", result["classification"]["intent"])
    print("Urgency:", result["classification"]["urgency"])
    print("Topic:", result["classification"]["topic"])
    print("Summary:", result["classification"]["summary"])
else:
    print("\nNo classification available")

print("="*60)
print("RESULT:")
print("="*60)
print("\nEmail Content:\n", result["email_content"])
print("\nCustomer Name:", result["sender_email"])
print("Customer Email ID:", result["email_id"])
print("\nDraft Response:\n", result["draft_response"])

# print("\nMessages:\n", result["messages"])
#-------------------------------------
# Visulize the graph
#-------------------------------------
# Get the graph in mermaid /PNG format and save it
graph_image = graph.get_graph(xray=True).draw_mermaid_png()

with open("Customer-Support-Agent.png", 'wb') as f:
    f.write(graph_image)

print("Graph Saved!")    