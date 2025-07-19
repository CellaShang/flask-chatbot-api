
from flask import Flask, request, jsonify
from pyngrok import ngrok
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import sqlite3

# ==== Load classification model and vectorizers ====
MODEL_PATH = '/content/drive/My Drive/Colab/AS4/STEP4-Champion_Clustering&Classification_Save&Evaluation/naive_bayes_bow_ngram.pkl'
BOW_VECTORIZER_PATH = '/content/drive/My Drive/Colab/AS4/STEP4-Champion_Clustering&Classification_Save&Evaluation/bow_vectorizer.pkl'
NGRAM_VECTORIZER_PATH = '/content/drive/My Drive/Colab/AS4/STEP4-Champion_Clustering&Classification_Save&Evaluation/ngram2_vectorizer.pkl'

classif_model = joblib.load(MODEL_PATH)
bow_vectorizer = joblib.load(BOW_VECTORIZER_PATH)
ngram_vectorizer = joblib.load(NGRAM_VECTORIZER_PATH)

label_map = {
    0: 'Gene Expression Analysis',
    1: 'Sequence Classification',
    2: 'Protein Structure Prediction',
    3: 'Biological Image Analysis',
    4: 'Disease Outcome Prediction'
}

def predict_paper_category(text):
    bow_features = bow_vectorizer.transform([text])
    ngram_features = ngram_vectorizer.transform([text])
    X = np.hstack([bow_features.toarray(), ngram_features.toarray()])
    label_num = classif_model.predict(X)[0]
    return label_map[label_num]

# ==== Load clustering model, PCA, vectorizer ====
output_dir = '/content/drive/My Drive/Colab/AS4/STEP4-Champion_Clustering&Classification_Save&Evaluation'

kmeans_model = joblib.load(os.path.join(output_dir, 'kmeans_tfidf_k5.pkl'))
pca_model = joblib.load(os.path.join(output_dir, 'pca_tfidf_100.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(output_dir, 'vectorizer_tfidf.pkl'))

def get_top_keywords_for_clusters(kmeans_model, pca_model, vectorizer, n_keywords=10):
    centers_pca = kmeans_model.cluster_centers_
    centers_tfidf = pca_model.inverse_transform(centers_pca)
    terms = vectorizer.get_feature_names_out()

    top_keywords = {}
    for i, center in enumerate(centers_tfidf):
        top_indices = center.argsort()[::-1][:n_keywords]
        keywords = [terms[idx] for idx in top_indices]
        top_keywords[i] = keywords
    return top_keywords

top_keywords_per_cluster = get_top_keywords_for_clusters(kmeans_model, pca_model, tfidf_vectorizer)

# ==== Load recommendation data and similarity matrices ====
df = pd.read_excel('/content/drive/My Drive/Colab/AS4/STEP4-Champion_Clustering&Classification_Save&Evaluation/Raw_with_predicted_classification_label.xlsx')
tfidf_matrix = load_npz('/content/drive/My Drive/Colab/AS4/STEP2-feature_Engineering/tfidf_matrix.npz')
metadata_matrix = load_npz('/content/drive/My Drive/Colab/AS4/STEP2-feature_Engineering/metadata_matrix.npz')

content_similarity = cosine_similarity(tfidf_matrix)
metadata_similarity = cosine_similarity(metadata_matrix)
final_similarity = 0.7 * content_similarity + 0.3 * metadata_similarity

def find_paper_index(title):
    matches = df.index[df['Article Title'].str.lower() == title.lower()]
    if len(matches) == 0:
        return None
    return matches[0]

def recommend_papers_by_index(paper_idx, top_n=10):
    sim_scores = final_similarity[paper_idx]
    top_indices = np.argsort(sim_scores)[::-1]
    top_indices = top_indices[top_indices != paper_idx][:top_n * 5]

    results = df.iloc[top_indices][['Article Title', 'Times Cited, All Databases', 'Author', 'Publication Year', 'Document Type', 'DOI Link']]
    results = results.copy()
    results['Similarity'] = sim_scores[top_indices]
    results = results.sort_values(by='Times Cited, All Databases', ascending=False).head(top_n)
    return results.reset_index(drop=True)

def summarize_lsa(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    if not summary:
        return text
    return ' '.join(str(sentence) for sentence in summary)

DATABASE = 'papers.db'
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)

@app.route('/predict_category', methods=['POST'])
def predict_category():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    category = predict_paper_category(text)
    return jsonify({'category': category})

@app.route('/cluster_predict', methods=['POST'])
def cluster_predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    X_vec = tfidf_vectorizer.transform([text])
    X_vec_reduced = pca_model.transform(X_vec.toarray())
    cluster_id = int(kmeans_model.predict(X_vec_reduced)[0])
    keywords = top_keywords_per_cluster.get(cluster_id, [])

    return jsonify({
        "cluster_id": cluster_id,
        "top_keywords": keywords
    })

@app.route('/cluster_themes', methods=['GET'])
def cluster_themes():
    safe_keywords = {int(k): [str(word) for word in v] for k, v in top_keywords_per_cluster.items()}
    return jsonify({
        "n_clusters": len(safe_keywords),
        "cluster_themes": safe_keywords
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    title = data.get('title', '').strip()
    top_n = int(data.get('top_n', 10))

    if not title:
        return jsonify({'error': 'Missing paper title'}), 400

    paper_idx = find_paper_index(title)
    if paper_idx is None:
        return jsonify({'error': f'Paper title "{title}" not found'}), 404

    recs_df = recommend_papers_by_index(paper_idx, top_n)
    recs = recs_df.to_dict(orient='records')

    return jsonify({
        'input_title': title,
        'recommendations': recs
    })

@app.route('/summarize_abstract', methods=['POST'])
def summarize_abstract():
    data = request.get_json(force=True)
    abstract = data.get('abstract', '').strip()
    if not abstract:
        return jsonify({'error': 'No abstract provided'}), 400

    num_sentences = int(data.get('num_sentences', 3))
    summary = summarize_lsa(abstract, num_sentences)
    return jsonify({'summary': summary})

@app.route('/summarize_by_title', methods=['POST'])
def summarize_by_title():
    data = request.get_json(force=True)
    title = data.get('title', '').strip()
    num_sentences = int(data.get('num_sentences', 3))

    if not title:
        return jsonify({'error': 'No paper title provided'}), 400

    matches = df.index[df['Article Title'].str.lower() == title.lower()]
    if len(matches) == 0:
        return jsonify({'error': f'Paper titled "{title}" not found'}), 404

    paper_idx = matches[0]
    abstract = df.loc[paper_idx, 'Abstract']
    if not abstract or pd.isna(abstract):
        return jsonify({'error': f'No abstract available for paper "{title}"'}), 404

    summary = summarize_lsa(abstract, num_sentences)
    return jsonify({
        'title': title,
        'summary': summary
    })

@app.route('/search_papers', methods=['GET'])
def search_papers():
    conn = get_db_connection()

    allowed_cols = {
        'id', 'article_title', 'abstract', 'author', 'publication_year',
        'document_type', 'keywords', 'times_cited', 'doi_link', 'predicted_classification_label'
    }

    fields = request.args.get('fields')
    if fields:
        requested_cols = [col.strip() for col in fields.split(',')]
        selected_cols = [col for col in requested_cols if col in allowed_cols]
        if not selected_cols:
            selected_cols = ['*']
    else:
        selected_cols = ['*']

    select_clause = ", ".join(selected_cols)

    query = f"SELECT {select_clause} FROM papers WHERE 1=1"
    params = []

    author = request.args.get('author')
    if author:
        query += " AND author LIKE ?"
        params.append(f"%{author}%")

    pub_year_min = request.args.get('year_min', type=int)
    if pub_year_min is not None:
        query += " AND publication_year >= ?"
        params.append(pub_year_min)

    pub_year_max = request.args.get('year_max', type=int)
    if pub_year_max is not None:
        query += " AND publication_year <= ?"
        params.append(pub_year_max)

    document_type = request.args.get('document_type')
    if document_type:
        query += " AND document_type = ?"
        params.append(document_type)

    classification = request.args.get('classification')
    if classification:
        query += " AND predicted_classification_label = ?"
        params.append(classification)

    keywords = request.args.get('keywords')
    if keywords:
        query += " AND abstract LIKE ?"
        params.append(f"%{keywords}%")

    order_by = request.args.get('order_by', default='times_cited')
    order_dir = request.args.get('order_dir', default='DESC').upper()
    if order_dir not in ['ASC', 'DESC']:
        order_dir = 'DESC'
    if order_by not in allowed_cols:
        order_by = 'times_cited'
    query += f" ORDER BY {order_by} {order_dir}"

    limit = request.args.get('limit', default=10, type=int)
    query += " LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = [dict(row) for row in rows]
    return jsonify({'papers': results})

if __name__ == '__main__':
    port = 5000
    public_url = ngrok.connect(port)
    print(f"🔗 ngrok tunnel URL: {public_url}")
    app.run(host='0.0.0.0', port=port)
