from openai import OpenAI
import base64

from dotenv import load_dotenv

def encode_image(image_path):
    with open(image_path,'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("./plot.png")

#client = OpenAI(a
input_messages = [
    {
        'role':'user',
        'content':[
            {
                "type": "input_text",
                "text": '''Provided is a user's speech in a presentation. Note that this was done through speech-to-text, so there may be grammatical inaccuracies, and also it was recorded in one big chunk; ignore grammar, punctuation in your feedback, focus on just the content. Provide one dot point of positive feedback (if any, be critical), and three fot points of negative feedback, returning just those 4 in a row (i.e.  - Feedback   - Feedback etc). Also note that the topic of the user's presentation is Robocup. User's speech:  you you you'''
            }
        ]
    }
]

response = client.responses.create(
    model='gpt-4o-mini',
    input=input_messages
)

print(response.output_text)