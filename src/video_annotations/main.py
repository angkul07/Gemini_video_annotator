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

GEMINI_API_KEY = "AIzaSyDeUg69r2i21sDFOqKuo_fVooxEcuGpknA"

client = genai.Client(api_key=GEMINI_API_KEY)

folder_path = Path("/home/angkul/my_data/coding/agents/video_save")

file_path = ["/home/angkul/my_data/coding/agents/video_save/" + f.name for f in folder_path.iterdir() if f.is_file()]

# video_file = "/home/angkul/my_data/coding/agents/video_save/2016-01-02_0700_US_KOCE_Tavis_Smiley_1291.38-1295.68_ago.mp4"
# print("Uploading file...")
# for video_file in file_path:
# video_file = client.files.upload(file="/home/angkul/my_data/coding/agents/video_save/2016-01-02_0700_US_KOCE_Tavis_Smiley_1291.38-1295.68_ago.mp4")
# print(f"Completed upload: {video_file.uri}")

# video_name = "US_KOCE_Tavis_Smiley_385.83-390.14_ago.mp4"


# Check whether the file is ready to be used.
# while video_file.state.name == "PROCESSING":
#     print('.', end='')
#     time.sleep(1)
#     video_file = client.files.get(name=video_file.name)

# if video_file.state.name == "FAILED":
#   raise ValueError(video_file.state.name)

# print('Done')

prompt = """
You are a annotations specialist. Your task is to find different annotations and information
in the given {video_file}. You should focus on finding the following annotations and answers in the given format:
[
  {
    "description": "Video_file name ",
    "value": {video_file}
  },
  {
    "description": "Is the person in the image standing?",
    "value": "standing"
  },
  {
    "description": "Are the person's hands visible?",
    "value": "hands_visible"
  },
  {
    "description": "Is the setting indoors or outdoors?",
    "value": "indoor"
  },
    "description": "The meaning of touch word in the video transcription has physical sense or Emotional sense?",
    "value": "physicaltouch"
  }
]
"""

result = []
for video_file in file_path:
  print("Uploading file...")
  video_file = client.files.upload(file=video_file)
  print(f"Completed upload: {video_file.uri}")

  while video_file.state.name == "PROCESSING":
      print('.', end='')
      time.sleep(1)
      video_file = client.files.get(name=video_file.name)

  if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
  print("Done")

  response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=[prompt, video_file]
    )
  result.append(response.text)

# print(response.text)
# user_input = response.text
print(result)

def run(user_input):
    """
    Run the crew.
    """
    inputs = {
        'JSON': user_input,
        # 'file_name': video_name
    }
    
    try:
        result = VideoAnnotations().crew().kickoff(inputs=inputs)
        # result = markdown.markdown(result)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
run(result)