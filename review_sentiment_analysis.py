import pandas as pd
import openai 
import os

from dotenv import load_dotenv

#reading csv
df = pd.read_csv("imbd_new.csv")

# #openAI api keys
openai.api_key = os.getenv('openAI')


'''
The code defines a function analyze_gpt35 to analyze the sentiment of a given text using the OpenAI GPT-3.5-turbo model.

Two roles are defined in the conversation: "system" providing instructions to the model, and "user" instructing the model to analyze 
the product review for sentiment.

Hyperparameters for the GPT-3.5 model, such as max_tokens, n (number of completions), stop (stopping criterion), and temperature are 
set based on the requirements.

The function is applied to each product review in the dataframe, and the predicted sentiment is stored in a new column called 
'predicted_gpt35'.

The comparison of actual sentiment and predicted sentiment using GPT-3.5 is displayed using the value_counts method on the relevant 
columns in the dataframe.
'''

# Function definition to analyze sentiment using the OpenAI GPT-3.5-turbo model.
def analyze_gpt35(text):
    # Define system and user roles in the conversation.
    messages = [
        {"role": "system", "content": """You are trained to analyze and detect the sentiment of given text. 
                                    If you're unsure of an answer, you can say "not sure" and recommend users to review manually."""},
        {"role": "user", "content": f"""Analyze the following product review and determine if the sentiment is: positive or negative. 
                                        Return the answer in a single word as either positive or negative: {text}"""}
    ]

    # Set hyperparameters for the GPT-3.5 model.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0
    )
    
    # Extract the sentiment analysis result from the model's response.
    response_text = response.choices[0].message.content.strip().lower()
    return response_text

# Calling GPT-3.5 to analyze sentiments for each product review in the dataframe.
df['predicted_gpt35'] = df['review'].apply(analyze_gpt35)

# Display the comparison of actual sentiment and predicted sentiment using GPT-3.5.
print(df[['sentiment', 'predicted_gpt35']].value_counts())
