# Install libraries

import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import time


# modify column width
pd.set_option("display.width", 320)
np.set_printoptions(linewidth=320)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display - was 10
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps
pd.set_option("display.max_colwidth", 60)

# Start timer
t0 = time.time()  # Add at start of process

# 1. Load Dataset
n_rows = 30                            # Define number of rows to import
df = pd.read_csv("books_1.Best_Books_Ever.csv", nrows=n_rows)  # Replace with actual file path
df = df[['title', 'author', 'description', 'genres']].dropna()
print(df.info())

# 2. Preprocess
# Filter out rows with certain descriptions
strings_to_remove = 'librarian|isbn'

mask_of_rows_to_keep = ~df['description'].str.contains(
    strings_to_remove,
    case=False,
    regex=True,
    na=False  # Treat missing/NaN values as not containing the string
)

df = df[mask_of_rows_to_keep]

# Strip out leading and trailing spaces and whitespace
df['description'] = df['description'].str.strip()
# df['genres_top'] = df['genres'].apply(lambda x: x.split(',')[0])  # Take first genre for simplicity
print(df)

# 3. Candidate Labels
level1_labels = ['Fiction', 'Non-Fiction']
fiction_labels = ["Science Fiction", "Adventure", "Romance", "Mystery", "Fantasy", "Historical"]
nonfiction_labels = ['Biography', 'History', 'Self-Help', 'Science', 'Business']


# 4. Zero-Shot Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# Apply the classifier and return the first label (which is that of highest probability
df['category'] = df['description'].apply(lambda row: classifier(row, level1_labels,
                                                  )['labels'][0])
df_output = df[['author', 'title', 'description', 'category']]
print(df_output)
df_output.to_excel('Book_category_output.xlsx', index=False)

# # Define predicted genre

# Define level 2 predicted genre
def classify_subgenre(description, category):
    if category == 'Fiction':
        labels = fiction_labels
    elif category == 'Non-Fiction':
        labels = nonfiction_labels
    else:
        return None
    result = classifier(description, labels)
    return result['labels'][0]

df['category_lv2'] = df.apply(lambda row: classify_subgenre(row['description'], row['category']), axis=1)
df_output2 = df[['author', 'title', 'description', 'category', 'category_lv2']]
print(df_output2)
df_output2.to_excel('Book_category_output2.xlsx', index=False)

# predictions = []
# true_labels = []
#
# for _, row in df.iterrows():  # Sample for speed
#     text = row['description']
#     true_label = row['genres_top']
#     result = classifier(text, candidate_labels)
#     predicted_label = result['labels'][0]  # Top prediction
#     predictions.append(predicted_label)
#     true_labels.append(true_label)
#
# # 5. Evaluation
# accuracy = accuracy_score(true_labels, predictions)
# precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0.0)
#
# print(f"Accuracy: {accuracy:.2f}")
#
#
# #
# import streamlit as st
# from transformers import pipeline
#
# # Load model
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# candidate_labels = ["Science Fiction", "Romance", "Mystery", "Fantasy", "Non-Fiction", "Historical"]
#
# st.title("Book Genre Predictor (Zero-Shot)")
# st.write("Enter a book description and get predicted genres:")
#
# description = st.text_area("Book Description", "")
# if st.button("Predict Genre"):
#     if description.strip():
#         result = classifier(description, candidate_labels)
#         st.subheader("Predicted Genres:")
#         for label, score in zip(result['labels'], result['scores']):
#             st.write(f"{label}: {score:.2f}")
#     else:
#         st.warning("Please enter a description.")
#

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"Time Taken: {timetaken1:.4f} seconds")
