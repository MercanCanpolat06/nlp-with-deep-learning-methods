### LINEAR TEXT CLASSIFIER ###
import ollama
import json
import time

words_to_label = []
vector_file = "../data/my_custom_vectors.txt"

print(f"reading {vector_file}")
with open(vector_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3000: break
        words_to_label.append(line.split()[0])

labeled_data = {}
batch_size = 50 

print(f"Labeling {len(words_to_label)} words with Qwen2.5:14b")

k = len(words_to_label)
for i in range(0, k, batch_size):
    batch = words_to_label[i : i + batch_size]

    prompt = f"""Task: Classify each word as VERB (1) or NOT A VERB (0).
Return ONLY a raw JSON object. No explanations.
Format: {{"word1": 1, "word2": 0}}

Words: {batch}"""

    try:
        response = ollama.chat(model='qwen2.5:14b', messages=[
            {'role': 'system', 'content': 'You are a linguistic expert. Output only valid JSON.'},
            {'role': 'user', 'content': prompt},
        ])
        
        content = response['message']['content']
        clean_content = content.replace('```json', '').replace('```', '').strip()
        
        batch_labels = json.loads(clean_content)
        labeled_data.update(batch_labels)
        
        print(f"Progress {len(labeled_data)} / {len(words_to_label)}")
        
    except Exception as e:
        print(f"Error (Batch {i}): {e}. Retrying")
        time.sleep(2)

with open("qwen_labels.json", "w", encoding="utf-8") as f:
    json.dump(labeled_data, f, ensure_ascii=False, indent=4)

print("\nDONE")