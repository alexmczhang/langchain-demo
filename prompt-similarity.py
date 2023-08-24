from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import VertexAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# These are a lot of examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
 
example_selector= SemanticSimilarityExampleSelector.from_examples(
    examples,
    VertexAIEmbeddings(),
    Chroma,
    k=1
)

similar_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:", 
    input_variables=["adjective"],
)

# Input is a feeling, so should select the windy/calm example
print(similar_prompt.format(adjective="Foggy"))

# Input is a measurement, so should select the tall/short example
print(similar_prompt.format(adjective="fat"))

# You can add new examples to the SemanticSimilarityExampleSelector as well
similar_prompt.example_selector.add_example({"input": "enthusiastic", "output": "apathetic"})
print(similar_prompt.format(adjective="joyful"))