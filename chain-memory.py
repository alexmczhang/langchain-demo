from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

# Text model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory() # k=2 , We set a low k=2, to only keep the last 2 interactions in memory     
)

print(conversation.predict(input="Hi there!"))

print(conversation.predict(input="I'm doing well! Just having a conversation with an AI."))

print(conversation.predict(input="Tell me about yourself."))