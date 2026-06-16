import os
import google.generativeai as genai

genai.configure(
    api_key="YOUR_API_KEY"
)
model = genai.GenerativeModel("gemini-2.0-flash")
def ask_gemini(context, question):

    response = model.generate_content(
        f"""
        Context:
        {context}

        Question:
        {question}

        Answer only from the context.
        """
    )

    return response.text