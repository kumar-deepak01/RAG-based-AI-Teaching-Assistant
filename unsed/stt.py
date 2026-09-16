import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe(audio= "audios/5_Exercise 1 - Pure HTML Media Player.mp3", 
                          language = "hi",
                          task ="translate",
                            word_timestamps=False)
print(result["text"])

# with open("output.json", "w") as f:
#     json.dump(result, f)
chunks = []
for segments in result["segments"]:
    chunks.append({"start": segments["start"], "end": segments["end"], "text": segments["text"]})

with open("output.json", "w") as f:
    json.dump(chunks,f)
    