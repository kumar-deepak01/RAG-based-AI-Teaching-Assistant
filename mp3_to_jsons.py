


import whisper
import json
import os

model = whisper.load_model("large-v2")

audios = os.listdir("audios")


for audio in audios:
    number = audio.split("_")[0]
    title = audio.split("_")[1][:-4]
    print(number, title)
    result = model.transcribe(audio=f"audios/{audio}", 
                          language="hi",
                          task="translate",
                          word_timestamps=False)

    chunks=[]

    for segments in result["segments"]:
        chunks.append({"number": number, "title": title, "start": segments["start"], "end": segments["end"], "text": segments["text"]})

    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}


    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)




