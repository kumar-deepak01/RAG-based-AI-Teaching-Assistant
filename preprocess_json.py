# import requests
# import os
# import json
# import pandas as pd

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# jsons = os.listdir("jsons")  # List all the jsons 
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
       
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk) 
# # print(my_dicts)

# df = pd.DataFrame.from_records(my_dicts)
# print(df)
# # a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# # print(a)
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


# jsons = sorted(os.listdir("jsons"))
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     if not json_file.endswith(".json"):
#         continue
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")

#     texts = [c['text'] for c in content['chunks']]
#     print(f"  chunk count: {len(texts)}")

#     embeddings = create_embedding(texts, batch_size=32)

#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk)

# df = pd.DataFrame.from_records(my_dicts)
# print(df)
# df.to_parquet("embeddings.parquet")
# print("Saved to embeddings.parquet")

df = pd.read_parquet("embeddings_fixed.parquet")
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

promt = f''' I am teaching web devlopment using sigma web development course. Here are video chunks containing video title, video number, start time in second, end time in second, the text at that time :

{new_df[["title", "number", "start", "end", "text"]].to_json()}
----------------------------------
"{incoming_query}"
user asked this quesion related to the video chunks , you have to answer where and how much content is taught where (in which video and at what timestamp) and guid the user to go to that particular video 
'''

with open("promt.text", "w") as f:
    f.write(promt)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["text"], item["start"], item["end"])

