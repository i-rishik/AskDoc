prompt_template = """
You are an expert at creating questions based on educational material.
Your goal is to prepare students for exams by generating insightful questions
from the text below:

----------
{text}
----------

Create questions that thoroughly test understanding of the content.
Ensure no key information is lost.

QUESTIONS:
"""

refine_template = """
You are an expert at refining questions for educational purposes.
We have some initial questions: {existing_answer}.
Use the additional context below to improve them:

----------
{text}
----------

Refine or expand the questions while ensuring clarity and relevance.
If the new context isn't useful, return the original questions.

QUESTIONS:
"""
