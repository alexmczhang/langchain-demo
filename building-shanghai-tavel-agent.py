import os
from google.cloud import aiplatform
aiplatform.init(project='hello-world-360207')

from langchain.chat_models import ChatVertexAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

llm = ChatVertexAI(temperature=0.5)

print("Tell me about the best 5 places in Shanghai\n")
print(llm.predict("tell me about the best five places in Shanghai"))

print("\nWhich museum in Shanghai can you recommend and how do I get there.\n")
print(llm.predict_messages([HumanMessage(content="Which museum in Shanghai can you recommend and how do I get there.")]))

# example to show there is no memory
print("\nWhat is my name?\n")
llm.predict_messages([HumanMessage(content="What is my name?")])

# Memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("Hi my name is Xiao-A")
memory.chat_memory.add_ai_message("Whats up Xiao-A?")
print(memory.buffer)

#Chain + Memory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

print("\nHi my name is Xiao-A provide me a list of top sightseeing trips in Shanghai\n")
print(conversation.predict(input="Hi my name is Xiao-A provide me a list of top sightseeing trips in Shanghai"))

print("\nyou remember my name?\n")
print(conversation.predict(input="you remember my name?"))

print("\nShow memory buffer from langchain\n")
print(memory.buffer)


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)

system_prompt = SystemMessagePromptTemplate.from_template("""
Act as a experienced Shanghai tour guide.
Answer friendly but if you get a question that is not related to Shanghai deny the answer with: "弄关特！".
""")
memory_prompt = MessagesPlaceholder(variable_name="history")
human_prompt = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    memory_prompt,
    human_prompt])

memory = ConversationBufferMemory(return_messages=True, memory_key="history",)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=chat_prompt
)

print(conversation.predict(input="Hi there!"))
print(conversation.predict(input="What can you recommend?"))
print(conversation.predict(input="We are in front of the Bund tell us something about it?"))
