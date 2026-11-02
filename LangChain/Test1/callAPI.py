import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in the .env file.")

model = ChatOpenRouter(
    model=os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free"),
    api_key=api_key,
    reasoning={"effort": "high", "summary": "auto"},
)

question = "How many r's are in the word 'strawberry'?"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer clearly and carefully."),
        ("human", "{question}"),
    ]
)
chain = prompt | model

first_response = chain.invoke({"question": question})
print("First answer:")
print(first_response.content)

# Continue the conversation using LangChain messages, including the model reply.
messages = [
    HumanMessage(content=question),
    first_response,
    HumanMessage(content="Are you sure? Think carefully."),
]

second_response = model.invoke(messages)
print("\nFollow-up answer:")
print(second_response.content)