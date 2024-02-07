category_template = """### Instruction:
Read the below input and determine if it is a request to search, explain, generate computer code?
Respond only with "generation" if it requests code, "explanation" if asks for explaination, "search" if it is searching for a function or tool and no other text.
Respond with "chat" if it does not fit any of the mentioned categories and no other text.</s>

### Input:
{query}</s>

### Response:
"""