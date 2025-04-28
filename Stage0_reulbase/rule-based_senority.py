import numpy as np
import re
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('punkt_tab')

# Train set
train_input_file_path = "./job_data_files/raw-data/seniority_labelled_development_set.csv"

# Test set
test_input_file_path = "./job_data_files/raw-data/seniority_labelled_test_set.csv"
output_file_path = "./job_data_files/df_output_seniority_rule_based_testset.csv"


df_senority = pd.read_csv(train_input_file_path)
df_senority_test = pd.read_csv(test_input_file_path)

a = set(df_senority['y_true'].unique())
b = set(df_senority_test['y_true'].unique())
label_dict = df_senority['y_true'].value_counts().to_dict()



labels = [
    "apprentice", "experienced", "intermediate", "senior", "director",
    "deputy", "chief", "doctoral", "assistant", "associate", "principal",
    "executive", "graduate", "entry level", "junior", "qualified"
]


model_path = 'GoogleNews-vectors-negative300.bin'  
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

label_embeddings = {}
for label in labels:
    words = label.split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
        else:
            print(f"Warning: 单词 '{word}' 不在词向量模型中。")
    if vectors:
        label_embeddings[label] = np.mean(vectors, axis=0)
    else:
        print(f"Warning: 标签 '{label}' 没有可用的词向量。")


def clean_text(text):
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def predict_label(text, model, label_embeddings, threshold=0.5):
    tokens = word_tokenize(text.lower())
    label_scores = {label: 0.0 for label in label_embeddings.keys()}
    
    for token in tokens:
        if token in model:
            token_vec = model[token]
            for label, emb in label_embeddings.items():
                sim = np.dot(token_vec, emb) / (np.linalg.norm(token_vec) * np.linalg.norm(emb))
                if sim > threshold:
                    label_scores[label] += sim

    total_score = sum(label_scores.values())
    if total_score > 0:
        prob_distribution = {label: score / total_score for label, score in label_scores.items()}
    else:
        prob_distribution = {label: 0 for label in label_scores.keys()}
    
    return prob_distribution

def get_predicted_label(row):
    combined_text = ' '.join([str(row.get(col, '')) for col in ['job_title', 'job_summary', 'job_ad_details']])
    cleaned_text = clean_text(combined_text)
    probs = predict_label(cleaned_text, model, label_embeddings, threshold=0.5)
    predicted = max(probs, key=probs.get)
    return predicted

df_senority['predicted'] = df_senority.apply(get_predicted_label, axis=1)
accuracy_dev = (df_senority['predicted'] == df_senority['y_true']).mean()
print(f"Accuracy: {accuracy_dev:.2%}")

df_senority_test['predicted'] = df_senority.apply(get_predicted_label, axis=1)
accuracy_test = (df_senority_test['predicted'] == df_senority_test['y_true']).mean()
print(f"Accuracy: {accuracy_test:.2%}")

# Keep only certain columns
df_senority_test = df_senority_test.rename(columns={'predicted': 'y_pred'})

df_senority_test = df_senority_test[['y_true', 'y_pred']]

df_senority_test.to_csv(output_file_path, index=False)

