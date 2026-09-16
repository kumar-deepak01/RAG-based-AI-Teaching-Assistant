import requests
import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list, batch_size=32, max_retries=3):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                r = requests.post("http://localhost:11434/api/embed", json={
                    "model": "bge-m3",
                    "input": batch
                }, timeout=120)
                data = r.json()
                if "embeddings" not in data:
                    print(f"  Batch {i}-{i+len(batch)} failed: {data.get('error', 'unknown error')}")
                    time.sleep(3)
                    continue
                all_embeddings.extend(data["embeddings"])
                break
            except requests.exceptions.RequestException as e:
                print(f"  Request exception on batch {i}: {e}")
                time.sleep(3)
        else:
            raise RuntimeError(f"Failed to embed batch starting at index {i} after {max_retries} retries")
    return all_embeddings

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })


    response=r.json()
    print(response)
    return response




df=pd.read_parquet("embeddings.parquet")
incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarities of question_embeddinf with other embeddings

# print(df['embedding'].values)
print(len(df['embedding'].iloc[0]))

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

top_result =5
best_idx = similarities.argsort()[::-1][0:top_result]
new_df = df.loc[best_idx]
# print(f"Best match index: {best_idx}, score: {similarities[best_idx]:.4f}")
# print(df.iloc[best_idx][['title','text']])    # ab yeh actual best match dega

promt = f''' I am teaching web devlopment in my sigma web development course. Here are video chunks containing video title, video number, start time in second, end time in second, the text at that time :

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
----------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("promt.text", "w") as f:
    f.write(promt)

response = inference(promt)["response"]
print(response)

with open("response.text", "w") as f:
    f.write(promt)
