import time
import base64
from crewai import Agent, Task, Crew
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path

from crew import VideoAnnotations

GEMINI_API_KEY = "<Your api key>"

client = genai.Client(api_key=GEMINI_API_KEY)


print("Uploading file...")
video_file = client.files.upload(file="/home/angkul/my_data/coding/agents/video_save/2016-01-02_0700_US_KOCE_Tavis_Smiley_1291.38-1295.68_ago.mp4")
print(f"Completed upload: {video_file.uri}")


# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(1)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

print('Done')

prompt = '''
You are a annotations specialist. Your task is to find different annotations and information
in the given {video_file}. You should focus on finding the following annotations and answers in the given format:
[
  {{
    "description": "Video_file name ",
    "value": "{video_file}"
  }},
  {{
    "description": "Is the person in the image standing?",
    "value": "standing"
  }},
  {{
    "description": "Are the person's hands visible?",
    "value": "hands_visible"
  }},
  {{
    "description": "Is the setting indoors or outdoors?",
    "value": "indoor"
  }},
  {{
    "description": "The meaning of touch word in the video transcription has physical sense or Emotional sense?",
    "value": "physicaltouch"
  }}
]
'''

response = client.models.generate_content(
  model="gemini-2.0-flash",
  contents=[prompt, video_file]
    )

print(response.text)
user_input = response.text

def run(user_input):
    """
    Run the crew.
    """
    inputs = {
        'JSON': user_input,
    }
    
    try:
        result = VideoAnnotations().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
run(user_input)