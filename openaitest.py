from openai import OpenAI
import base64

from dotenv import load_dotenv

def encode_image(image_path):
    with open(image_path,'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("./plot.png")

client = OpenAI(api_key = "My Key")

input_messages = [
    {
        'role':'user',
        'content':[
            {
                "type": "input_text",
                "text": "This graph shows the positions and movement of two of a user's wrists. Anytime a hand goes drastically up, it means they have started a hand gesture. when their hand goes drastically down, it means they have ended that gesture. give me two lists, one for each hand, with paris of values within denoting the start and end of every gesture done. The gestures should be substantial; eg from frame 21 to 29 is not a gesture, it is just normal hand swaying."
            },
            {
                "type":"input_image",
                "image_url":f"data:image/jpeg;base64,{base64_image}"
            }
        ]
    }
]

response = client.responses.create(
    model='gpt-4o-mini',
    input=input_messages
)

print(response.output_text)