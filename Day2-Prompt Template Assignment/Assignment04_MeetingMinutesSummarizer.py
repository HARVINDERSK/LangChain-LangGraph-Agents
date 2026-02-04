from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOllama(model = "gemma2:2b")

system_message ="""
 Role: You are an expert with skills to summarize meeting minutes

 Task: Summarize raw meeting transcript into decisons and action items with owners and deadlines.
 
 Rules: 
 - Structured Markdowns and include confidence score with each item.
 - Meeting minutes must be based on the transcript provided
 - Avoid adding items not in the transcript
 - ensure clarity and accuracy

 Response format:
 - markdown: ## Descisions, ## Action Items (with Owners,deadlines and confidence score)

 user question: {transcript}"""

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template("{transcript}")    
])
user_name = input("Hey, may i know you name please: ")

transcript = """
Meeting Title: Weekly Project Sync
Date: March 12, 2026
Time: 10:00 AM – 10:30 AM
Participants: Alex, Jordan, Priya, Sam

10:00 AM Alex: Good morning, everyone. Let’s get started. Thanks for joining on time.

10:01 AM Jordan: Morning.

10:01 AM Priya: Hi everyone.

10:02 AM Sam: Morning.

10:02 AM Alex: The goal today is to review progress on the Q2 launch and flag any blockers. Jordan, do you want to start with engineering updates?

10:03 AM Jordan: Sure. The core functionality is complete. We’re still working through some performance issues, but nothing critical at the moment.

10:05 AM Alex: Do you expect those issues to impact the timeline?

10:05 AM Jordan: No, I don’t think so. We should have them resolved by early next week.

10:06 AM Alex: Great. Priya, how are things on the design side?

10:06 AM Priya: Design is finalized and handed off. We’re just waiting on confirmation for the last accessibility review.

10:08 AM Sam: From QA’s side, we’ve started testing the latest build. We’ve logged a few minor bugs, but no major concerns so far.

10:09 AM Alex: Thanks, Sam. Are those bugs already in the tracker?

10:09 AM Sam: Yes, all documented and assigned.

10:10 AM Alex: Perfect. Any cross-team dependencies we should be aware of?

10:11 AM Jordan: We’ll need final copy from marketing before we lock the onboarding flow.

10:12 AM Alex: Noted. I’ll follow up with marketing today.

10:13 AM Priya: One more thing — we might need a quick review meeting once the accessibility feedback comes in.

10:14 AM Alex: Sounds good. Let’s tentatively plan for that later this week.

10:15 AM Sam: Works for me.

10:16 AM Alex: Any other questions or concerns?

10:17 AM (silence)

10:18 AM Alex: Alright, action items are: Jordan to resolve performance issues, Sam to continue QA, Priya to coordinate accessibility feedback, and I’ll follow up with marketing. Thanks everyone.

10:19 AM Jordan: Thanks.

10:19 AM Priya: Thanks.

10:19 AM Sam: Thank you.

10:20 AM Meeting ended."""

# while True:
# transcript = input("Enter your meeting transcript: ")
response = llm.invoke(template.format(
                user_name = user_name,
                transcript = transcript
    ))

print(f"Assistant : {response.text}")