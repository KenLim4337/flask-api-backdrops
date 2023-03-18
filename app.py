from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import json
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys
from datetime import datetime

app = Flask(__name__)

# Put this on environment variable
s3_bucket = os.environ['S3_ENDPOINT']

parquet_folder = f'{s3_bucket}face_data_app.parquet'

print(f'[{datetime.now().time()}]: Initializing')

pa.set_cpu_count(8)
pa.set_io_thread_count(8)

print(f'[{datetime.now().time()}]: Loading parquet files...')


df = pq.read_table(parquet_folder, columns=['photo_id', 'mae_embeddings'], use_threads=True, memory_map=True, pre_buffer=True).to_pandas()
df

path_to_models = os.path.join(app.root_path, 'models')

print(f'[{datetime.now().time()}]: Loading models...')

neigh_MAE = pickle.load(open(f'{path_to_models}/neigh_MAE.pkl', 'rb'))

clf = LogisticRegression(class_weight='balanced', random_state=1, max_iter=100000)

print(f'[{datetime.now().time()}]: Ready to accept requests')

@app.route('/')
def hello_world():
    return 'This is my first API call!'


@app.route('/api/search_embedding/', methods=["POST"])
def nn_search():
    query_id = request.json['query_id']
    query_embedding = df[df['photo_id'] == query_id]['mae_embeddings'].values[0]
    neighbours = neigh_MAE.kneighbors([query_embedding])
    indexes = neighbours[1][0]
    photo_ids = df.iloc[indexes]['photo_id'].values.tolist()

    distances = 1 - neighbours[0][0]
    similarities = list(zip(photo_ids, distances.tolist()))

    return jsonify(similarities)


@app.route('/api/build_classifier/', methods=["POST"])
def build_classifier():
    photo_id = request.json['photo_id']
    similar = json.loads(request.json['similar'])
    different = json.loads(request.json['different'])
    result_ids = json.loads(request.json['result_ids'])
    soft_negatives = json.loads(request.json['soft_negatives'])

    positive_embeddings = []
    negative_embeddings = []

    positive_embeddings = df[df['photo_id'].isin(similar)]['mae_embeddings'].values.tolist()
    positive_labels = [1] * len(positive_embeddings)
    positive_weights = [1] * len(positive_embeddings)

    negative_embeddings = df[df['photo_id'].isin(different)]['mae_embeddings'].values.tolist()
    negative_labels = [0] * len(negative_embeddings)
    negative_weights = [.1] * len(negative_embeddings)

    search_results = df[df['photo_id'].isin(result_ids)]
    soft_negatives = search_results[search_results['photo_id'].isin(soft_negatives)]
    padding_embeddings = soft_negatives['mae_embeddings'].values.tolist()
    padding_labels = [0] * len(padding_embeddings)
    padding_weights = [.1] * len(padding_embeddings)

    training_samples = positive_embeddings + negative_embeddings + padding_embeddings
    sample_weights = positive_weights + negative_weights + padding_weights
    training_labels = positive_labels + negative_labels + padding_labels

    clf.fit(training_samples, training_labels, sample_weights)

    updated_results = []

    for idx, row in search_results.iterrows():
        embedding = row['mae_embeddings']
        prediction = clf.predict_proba([embedding])[0][0]
        updated_results.append([row['photo_id'], prediction])

    updated_results = sorted(updated_results, key=lambda x: x[1])
    updated_results = [(x[0], x[1].item()) for x in updated_results if x[0] != photo_id]

    return jsonify(updated_results)


@app.route('/api/extract_embeddings/', methods=["POST"])
def extract_embeddings():
    photo_id = request.json['image_link']

    return jsonify(photo_id)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)