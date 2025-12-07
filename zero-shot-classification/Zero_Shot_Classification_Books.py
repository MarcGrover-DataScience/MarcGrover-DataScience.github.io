# Install libraries

import pandas as pd
from transformers import pipeline
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
n_rows = 100                            # Define number of rows to import
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
# level1_labels = ['Fiction', 'Non-Fiction']
# fiction_labels = ["Science Fiction", "Adventure", "Romance", "Mystery", "Fantasy", "Historical"]
# nonfiction_labels = ['Biography', 'History', 'Self-Help', 'Science', 'Business']

level1_labels = ["Science Fiction", "Adventure", "Romance", "Mystery", "Fantasy", "Historical", "Biography"]
# level2_labels = ["Children", "Young Adult", "Adult"]


# 4. Zero-Shot Classification
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=-1)
# facebook/bart-large-mnli    roberta-large-mnli

# Apply the classifier predictor and return the first label (which is that of highest probability)
df['category'] = df['description'].apply(lambda row: classifier(row, level1_labels,
                                                  )['labels'][0])
# # The following line is to classify the target audience based on description
# # This code was removed due to the poor output - based on human validation
# df['category2'] = df['description'].apply(lambda row: classifier(row, level2_labels,
#                                                   )['labels'][0])
df_output = df[['author', 'title', 'description', 'category']]
print(df_output)
df_output.to_excel('Book_category_output_roberta.xlsx', index=False)


# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"Time Taken: {timetaken1:.4f} seconds")
