import pandas as pd
import openai
import os
from dotenv import load_dotenv

#loading env files which contain my credentials
load_dotenv()

#initialising openAI key
openai.api_key = os.getenv('openAI')

'''
The function completegpt is defined to utilize the OpenAI GPT-3.5-turbo model for text completion.

The messages list includes system and user roles, providing context for the model.

The openai.ChatCompletion.create method is used to interact with the model, specifying parameters such as the model name, messages, 
temperature, number of completions (n), and stopping criterion (stop).

The temperature parameter is set to 0 for precision and reduced creativity.

The completed text is extracted from the model's response and returned by the function.

An example usage of the function is provided, completing the text "I love eating mangoes in" and printing the result.
'''

# Function definition for completing text using the OpenAI GPT-3.5-turbo model.
def completegpt(text):
    # Define system and user roles in the conversation.
    messages = [
        {"role": "system", "content": """you are trained to analyze the text and complete the text based on the semantic understanding of text"""},
        {"role": "user", "content": f"""Analyze the following text and based on the semantic understanding complete the text.
                                    Return the answer by completing the text: {text}"""}
    ]

    # Use the OpenAI GPT-3.5-turbo model to generate a completion for the given text.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,  # Temperature set to 0 for precision and reduced creativity.
        n=4,  # Number of completions to generate.
        stop=None  # No specific stopping criterion for the generated completions.
    )

    # Extract the completed text from the model's response.
    response_text = response.choices[0].message.content.strip().lower()
    return response_text

# Example usage of the function with a prompt.
completed_text = completegpt("I love eating mangoes in")
print(completed_text)


