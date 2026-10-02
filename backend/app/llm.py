import os
from dotenv import load_dotenv 
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROG_API_KEY")
)

def generate_answer(question: str, context: list):
    context_text = "\n\n".join(
        item["content"]
        for item in context
    )

    prompt = f"""
You are DocuMind, a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided in the document context below.

IMPORTANT RULES:
1. Never make up information.
2. Use only information explicitly present in the document context.
3. Answer the user's specific question directly and concisely.
4. Do not add information from you general knowlegde.

PROJECT RULES:

5. If the user asks about PROJECTS, only include items that are explicitly identified as projects in the document.

6. DO NOT treat the following as projects unless the document explicitly identifies them as projects:
- courses 
- certifications
- certificates
- trainig program
- skills 
- educations
- internships

7. For project-related questions, pay special attention to sections such as:
- PROJECT DETAILS
- PROJECT Title
- PROJECT Synopsis
- Projects

8. If a project title and description are available, provide the project title and a short summary of what was done.

9. If multiple projects are explicitly identified, list all of them that are supported by the provided context.

10. Do not call courses, certifications, or training programs "projects" just beacuse they appear near the PROJECT DETAILS section.

11. If the required information cannot be found in the provided context, say exactly:
"I couldn't find that information in the uploaded document."

DOCUMENT_CONTEXT:
{context_text}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature = 0
    )

    return response.choices[0].message.content 