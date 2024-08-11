import os  # Module for interacting with the operating system
from sklearn.feature_extraction.text import TfidfVectorizer  # Module for text vectorization using TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # Module for calculating cosine similarity

# Get a list of all text files in the current directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Read the contents of each student's text file
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Function to vectorize the text using TF-IDF
def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

# Function to calculate cosine similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])[0][1]  # Return the similarity score directly

# Vectorize the student notes using TF-IDF
vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

# Function to check plagiarism among the student notes
def check_plagiarism():
    global s_vectors
    for i, (student_a, text_vector_a) in enumerate(s_vectors):
        for j, (student_b, text_vector_b) in enumerate(s_vectors):
            if i != j:  # Ensure we are not comparing the same document
                # Calculate cosine similarity between two text vectors
                sim_score = similarity(text_vector_a, text_vector_b)
                # Sort the student file names alphabetically to avoid duplicates
                student_pair = tuple(sorted((student_a, student_b)))
                # Create a tuple with student file names and similarity score
                score = (student_pair[0], student_pair[1], sim_score)
                # Add the tuple to plagiarism_results set
                plagiarism_results.add(score)
    return plagiarism_results

# Print the plagiarism results
for data in check_plagiarism():
    print("Similarity data:\n", data)
