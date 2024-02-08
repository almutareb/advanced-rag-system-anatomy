# Trulens Evaluation
# Import main tools
from trulens_eval import TruChain, Feedback, Huggingface, Tru
from trulens_eval.schema import FeedbackResult
from trulens_eval.feedback.provider import OpenAI
import numpy as np
from core_langchain_rag import qa


# create a feedback function
tru = Tru()
tru.reset_database()
# Initialize HuggingFace-based feedback function collection class:
hugs = Huggingface()
openai = OpenAI()

# Define a language match feedback function using HuggingFace.
lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# Question/answer relevance between overall question and answer.
qa_relevance = Feedback(openai.relevance).on_input_output()
# By default this will evaluate feedback on main app input and main app output.

# Toxicity of input
#toxicity = Feedback(openai.toxicity).on_input()

# wrap your chain with TruChain
truchain = TruChain(
    qa,
    app_id='Chain1_ChatApplication',
    feedbacks=[lang_match, qa_relevance]
)
truchain("write an agent that searches wikipedia")

tru.run_dashboard() # open a Streamlit app to explore