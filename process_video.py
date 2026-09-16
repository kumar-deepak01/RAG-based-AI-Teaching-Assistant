# convert the video mp3 

import os
import subprocess 

files = os.listdir("Lectures")
for tutorial_number, file in enumerate(files, start=1):
    # print(index, file)
    file_name = file.split(" _")[0]
    print( tutorial_number, file_name)
    subprocess.run(["ffmpeg", "-i", f"Lectures/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])


