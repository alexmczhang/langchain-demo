from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users questions as best as possbile.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

# Text model
model = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

_input = prompt.format_prompt(question="What is the most famous city in China")
output = model(_input.to_string())

print(output_parser.parse(output))

# Chat model
chat_model = VertexAI(
    model_name="chat-bison@001",
    max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("answer the users questions as best as possbile.\n{format_instructions}\n{question}")
    ],
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)
_input = prompt.format_prompt(question="What is the most famous city in China")
output = chat_model(_input.to_messages())
print(output_parser.parse(output))