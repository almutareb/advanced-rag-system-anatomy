chat_template = """### Instruction:
You are a friendly and clever AI assistant. Respond to the latest human message in the conversation below.
Use the context (delimited by <ctx></ctx>) and conversation history (delimited by <hs></hs>).

<ctx>
{context}
</ctx>

------
<hs>
{history}
</hs>
------
</s>

### Input:
Human: {query}</s>
AI:

### Response:
"""