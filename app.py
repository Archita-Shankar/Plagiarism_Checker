from flask import Flask, render_template, request, redirect, url_for
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Function to vectorize the text using TF-IDF
def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

# Function to calculate cosine similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])[0][1]

# Function to check plagiarism among the student notes
def check_plagiarism(student_files, student_notes):
    vectors = vectorize(student_notes)
    s_vectors = list(zip(student_files, vectors))
    plagiarism_results = set()
    for i, (student_a, text_vector_a) in enumerate(s_vectors):
        for j, (student_b, text_vector_b) in enumerate(s_vectors):
            if i != j:
                sim_score = similarity(text_vector_a, text_vector_b)
                student_pair = tuple(sorted((student_a, student_b)))
                score = (student_pair[0], student_pair[1], sim_score)
                plagiarism_results.add(score)
    return plagiarism_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        student_files = [file.filename for file in files]
        student_notes = [file.read().decode('utf-8') for file in files]

        results = check_plagiarism(student_files, student_notes)
        return render_template('index.html', results=results)
    
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)
