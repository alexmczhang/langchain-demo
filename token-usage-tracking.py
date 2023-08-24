from google.cloud import aiplatform
from langchain.llms import VertexAI
# from langchain.callbacks import get_vertexai_callback

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

llm("Show me a joke")
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)
print(len(llm_result.generations))
print(llm_result.generations[0])
print(llm_result.generations[-1])

print("\n llm_resutl.llm_output")
print(llm_result.llm_output)

print("How many tokens for [what a joke]")
print(llm.get_num_tokens("what a joke"))


print("How many tokens for [What's the weather in Shanghai?]")
print(llm.get_num_tokens("What's the weather in Shanghai?"))