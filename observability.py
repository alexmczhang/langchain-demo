# Dependencies for usage example.
from observabilityhelper import OutputFormatter, AllChainDetails
from langchain import LLMChain, PromptTemplate
from langchain.llms import VertexAI
import vertexai  # Comes from google-cloud-aiplatform package.
from prettyprinter import cpprint

PROJECT_ID = "hello-world-360207" #  Put your project ID here.
REGION = "us-central1"

# Initiaize connection to Vertex PaLM API LLM.
vertexai.init(project=PROJECT_ID, location=REGION)
llm = VertexAI(temperature=0)


# Callback handler specified at execution time, more information given.
prompt_template = "What food pairs well with {food}?"
handler = AllChainDetails()
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
cpprint(llm_chain("chocolate", callbacks=[handler]))

print("\n---------------------------------------------------\n")

# Callback handler specified at initialization, less information given.
prompt_template = "What food pairs well with {food}?"
handler = AllChainDetails()
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    callbacks=[handler])
cpprint(llm_chain("chocolate"))

from observabilityhelper import OutputFormatter, AllChainDetails
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
import wikipedia

llm = VertexAI(temperature=0)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
handler = AllChainDetails()

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
cpprint(
    agent.run("Who was the president of the United States when Jordan won his first NBA playoffs championship？"
    , callbacks=[handler]))

#agent = initialize_agent(
#    tools,
#    llm,
#    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#    verbose=True)
#cpprint(agent.run("What former US President had a chimp costar?"))