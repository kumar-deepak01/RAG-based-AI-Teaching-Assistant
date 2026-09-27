# import requests
# import os
# import json
# import time
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def create_embedding(text_list, batch_size=32, max_retries=3):
#     all_embeddings = []
#     for i in range(0, len(text_list), batch_size):
#         batch = text_list[i:i + batch_size]
#         for attempt in range(max_retries):
#             try:
#                 r = requests.post("http://localhost:11434/api/embed", json={
#                     "model": "bge-m3",
#                     "input": batch
#                 }, timeout=120)
#                 data = r.json()
#                 if "embeddings" not in data:
#                     print(f"  Batch {i}-{i+len(batch)} failed: {data.get('error', 'unknown error')}")
#                     time.sleep(3)
#                     continue
#                 all_embeddings.extend(data["embeddings"])
#                 break
#             except requests.exceptions.RequestException as e:
#                 print(f"  Request exception on batch {i}: {e}")
#                 time.sleep(3)
#         else:
#             raise RuntimeError(f"Failed to embed batch starting at index {i} after {max_retries} retries")
#     return all_embeddings

# def inference(prompt):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         # "model": "deepseek-r1",
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream": False
#     })


#     response=r.json()
#     print(response)
#     return response



######################################################
# df = pd.read_parquet("embeddings_fixed.parquet").reset_index(drop=True)
# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0]

###################################################################3

# print(question_embedding)

# Find similarities of question_embeddinf with other embeddings

# print(df['embedding'].values)
############################################
# print(len(df['embedding'].iloc[0]))

# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# ###################################################3

# {top_result =5
# best_idx = similarities.argsort()[::-1][0:top_result]
# new_df = df.iloc[best_idx]} replace this code 

#######################################################

# TOP_K = 5
# THRESHOLD = 0.45

# sorted_idx = similarities.argsort()[::-1]
# filtered_idx = [i for i in sorted_idx if similarities[i] >= THRESHOLD][:TOP_K]

# if not filtered_idx:
#     print("Sorry, is question ke liye relevant content nahi mila course me.")
#     new_df = df.iloc[0:0]  # empty dataframe, taaki aage crash na ho
# else:
#     new_df = df.iloc[filtered_idx].copy()
#     new_df["score"] = similarities[filtered_idx]
#     print(new_df[["title", "number", "score"]])  # debug ke liye dekh lo ranking sahi hai ya nahi

###########################################################################################

# print(f"Best match index: {best_idx}, score: {similarities[best_idx]:.4f}")
# print(df.iloc[best_idx][['title','text']])    # ab yeh actual best match dega

###### promt = f''' I am teaching web devlopment in my sigma web development course. Here are video chunks containing video title, video number, start time in second, end time in second, the text at that time :

# {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
# ----------------------------------
# "{incoming_query}"
# User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
# '''

# with open("promt.text", "w") as f:
#     f.write(promt)

# response = inference(promt)["response"]
# print(response)

# with open("response.text", "w") as f:
#     f.write(promt)

############################################################


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
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("INFERENCE ERROR:", e)
        raise

def sec_to_mmss(sec):
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"

# ---- Load fixed/clean embeddings file ----
df = pd.read_parquet("embeddings_fixed.parquet").reset_index(drop=True)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

TOP_K = 5
THRESHOLD = 0.45

sorted_idx = similarities.argsort()[::-1]
filtered_idx = [i for i in sorted_idx if similarities[i] >= THRESHOLD][:TOP_K]

if not filtered_idx:
    print("Sorry, is question ke liye relevant content nahi mila course me.")
else:
    new_df = df.iloc[filtered_idx].copy()
    new_df["score"] = similarities[filtered_idx]
    print(new_df[["title", "number", "start", "end", "score"]].to_string())

    # Har result ke liye mm:ss timestamp bana lo
    new_df["start_mmss"] = new_df["start"].apply(sec_to_mmss)

    # LLM ko sirf transcript text dikhao (number/timestamp nahi), taaki explanation likhe
    chunks_for_llm = "\n".join(
        f"{i+1}. \"{row['text']}\"" for i, (_, row) in enumerate(new_df.iterrows())
    )

    promt = f'''You are a friendly teaching assistant. A student asked: "{incoming_query}"

Here are transcript lines from the course, most relevant first:
{chunks_for_llm}

Write a short, friendly explanation (3-5 sentences) covering what these lines teach, in the context of the student's question. Refer to them as "the first result", "the second result", etc. if needed.
DO NOT mention any video number or timestamp — those will be listed separately below your explanation.'''

    with open("promt.text", "w", encoding="utf-8") as f:
        f.write(promt)

    result = inference(promt)
    explanation = result.get("response", "").strip()

    # Sources list Python se banao - 100% accurate numbers/timestamps
    sources_list = "\n".join(
        f"{i+1}. 📺 Video {row['number']} ({row['title']}) — ⏱️ {row['start_mmss']}"
        for i, (_, row) in enumerate(new_df.iterrows())
    )

    final_answer = (
        f"{explanation}\n\n"
        f"Related spots in the course:\n"
        f"{sources_list}"
    )

    print("\n--- ANSWER ---")
    print(final_answer)

    with open("response.text", "w", encoding="utf-8") as f:
        f.write(final_answer)