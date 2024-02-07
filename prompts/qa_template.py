qa_template = """### Instruction:
Read the documentation exerpt (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) below to respond with a detailed answer to the given question.
First take a step back to reflect on the question breaking it down into smaller steps and then go through the answer step by step and one function at a time.
If the question cannot be answered based on the documentation exerpt alone or the documentation does not explicitly say the answer to the question,
write "Sorry I had trouble answering this question, based on the information I found."

<ctx>
Documentation: {context}
</ctx>

------
<hs>
{history}
</hs>
------
</s>

### Input:
Question: {query}</s>

### Response:
"""