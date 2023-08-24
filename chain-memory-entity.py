from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory

# Text model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

memory = ConversationEntityMemory(llm=llm, return_message=True)
_input = {"input": "Deven & Sam are working on a startup project regarding Generative AI"}
memory.load_memory_variables(_input)
memory.save_context(
    _input,
    {"output": " That sounds like a great project! What kind of project are they working on?"}
)

print(memory.load_memory_variables({"input": 'who is Sam'}))

memory = ConversationEntityMemory(llm=llm, return_messages=True)
_input = {"input": "Deven & Sam are working on a hackathon project"}
memory.load_memory_variables(_input)
memory.save_context(
    _input,
    {"ouput": " That sounds like a great project! What kind of project are they working on?"}
)

print(memory.load_memory_variables({"input": 'who is Sam'}))

