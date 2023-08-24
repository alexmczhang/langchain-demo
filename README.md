# langchain-demo
self-learning for langchian with Vertex AI by Google Cloud


# Setup your env
pip install langchain
pip install langchain[llms]
pip install google-cloud-aiplatform

# source file description

- building-shanghai-tavel-agent, # build a travel agent with Memory, Chain and ConversationChain
- chain-memory-entity.py, # text-bision with Memory
- chain-converscation.py, # SemanticSimilarityExampleSelector will select examples that are similar to your input by semantic meaning
- observability.py, # Agent + Chain to solve complex problem, the result is better and zero shot
- observabilityhelper.py, # implement BaseCallbackHandler to logging and monitor langchain module behavior
