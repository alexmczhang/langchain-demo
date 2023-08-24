# Utils
import time
from typing import List

# Langchain
import langchain
from pydantic import BaseModel

print(f"LangChain version: {langchain.__version__}")

# Vertex AI
from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.schema import HumanMessage, SystemMessage

print(f"Vertex AI SDK version: {aiplatform.__version__}")

# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]
    
# LLM model
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# Chat
chat = ChatVertexAI()

# Text
# You'll be working with simple strings (that'll soon grow in complexity!)
my_text = "What day comes after Friday?"

response = llm(my_text)
print(response)

# Chat Messages
chat([HumanMessage(content="Hello")])

res = chat(
    [
        SystemMessage(
            content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"
        ),
        HumanMessage(content="I like beef, what should I eat?"),
    ]
)

print(res.content)

res = chat(
    [
        HumanMessage(
            content="What are the ingredients required for making a beef burger?"
        )
    ]
)
print(res.content)
res = chat([HumanMessage(content="How many slices of bread you said?")])
print("How many slices of bread you said? " + res.content)

# Documents
from langchain.schema import Document

Document(
    page_content="This is my document. It is full of text that I've gathered from other places",
    metadata={
        "my_document_id": 234234,
        "my_document_source": "The LangChain Papers",
        "my_document_create_time": 1680013019,
    },
)

llm("What day comes after Friday?")


res = chat(
    [
        SystemMessage(content="You are a helpful AI bot to figure out travel plans."),
        HumanMessage(content="I would like to go to Shanghai, how should I do this?"),
    ]
)
print(res.content)

# Text Embedding Model

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

text = "Hi! It's time for the beach"
text_embedding = embeddings.embed_query(text)
print(f"Your embedding is length {len(text_embedding)}")
print(f"Here's a sample: {text_embedding[:5]}...")

#Prompts
prompt = """
Today is Monday, tomorrow is Wednesday.

What is wrong with that statement?
"""

llm(prompt)

from langchain import PromptTemplate

# Notice "location" below, that is a placeholder for another value later
template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location="Beijing")

print(f"\nFinal Prompt: {final_prompt}")
print("-----------")
print(f"LLM Output: {llm(final_prompt)}")

# Example Selectors
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS 
#pip3 install faiss-gpu

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

# Examples of locations that nouns are found
examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]


# SemanticSimilarityExampleSelector will select examples that are similar to your input by semantic meaning

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    embeddings,
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k=2,
)

similar_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    # Your prompt
    example_prompt=example_prompt,
    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {noun}\nOutput:",
    # What inputs your prompt will receive
    input_variables=["noun"],
)


# Select a noun!
my_noun = "football player"
print(similar_prompt.format(noun=my_noun))
print(llm(similar_prompt.format(noun=my_noun)))
print("\n")

# Output Parsers
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# How you would like your reponse structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(
        name="bad_string", description="This a poorly formatted user input string"
    ),
    ResponseSchema(
        name="good_string", description="This is your response, a reformatted response"
    ),
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# See the prompt template you created for formatting
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly including country, city and state names

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template,
)

promptValue = prompt.format(user_input="welcom to dbln!")

print(promptValue)

llm_output = llm(promptValue)
print("\n Raw llm output" + llm_output)
print("\n parsered llm output \n")
print(output_parser.parse(llm_output))

# Document Loaders
# pip3 install BeautifulSoup4
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# Ingest PDF files
from langchain.document_loaders import PyPDFLoader

loader = WebBaseLoader("http://www.paulgraham.com/worked.html")
data = loader.load()
print(f"Found {len(data)} comments")
print(f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:2]])}")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
)

texts = text_splitter.split_documents(data)

print(f"You have {len(texts)} documents")

print("Preview:")
print(texts[0].page_content, "\n")
print(texts[1].page_content)

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(data)

# Embed your texts
db = FAISS.from_documents(texts, embeddings)

# Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()

print("\n retriever")
print(retriever)

docs = retriever.get_relevant_documents(
    "what types of things did the author want to develop or build?"
)

print("\n\nwhat types of things did the author want to develop or build?")
print("\n\n".join([x.page_content[:200] for x in docs[:2]]))

# Memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")
conversation.predict(input="What is the capital of France?")
conversation.predict(input="What are some popular places I can see in France?")
conversation.predict(input="What question did I ask first?")

# Chain
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=llm, prompt=prompt_template)


template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=llm, prompt=prompt_template)

overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)

review = overall_chain.run("Shanghai")
print("\n\n Chain Demo ")
print(review)


# Summarization Chain
loader = WebBaseLoader(
    "https://cloud.google.com/blog/products/ai-machine-learning/how-to-use-grounding-for-your-llms-with-text-embeddings"
)
documents = loader.load()

print(f"# of words in the document = {len(documents[0].page_content)}")

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts)



# Load GOOG's 10K annual report (92 pages).
url = "https://abc.xyz/investor/static/pdf/20230203_alphabet_10K.pdf"
loader = PyPDFLoader(url)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f"# of documents = {len(docs)}")

# select embedding engine - we use Vertex PaLM Embeddings API
print(embeddings)

# Store docs in local vectorstore as index
# it may take a while since API is rate limited
from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs, embeddings)
# Expose index to the retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Create chain to answer questions
from langchain.chains import RetrievalQA

# Uses LLM to synthesize results from the search index.
# We use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

query = "What was Alphabet's net income in 2022?"
result = qa({"query": query})
print(result)

query = "How much office space reduction took place in 2023?"
result = qa({"query": query})
print(result)

# pip3 install faiss-gpu
# pip3 install BeautifulSoup4
# pip3 install pypdf
# pip3 install chromadb==0.3.29

# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/examples/langchain-intro/intro_langchain_palm_api.ipynb