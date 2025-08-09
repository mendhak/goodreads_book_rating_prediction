import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Load the ratings data (6M rows)
print("Loading ratings data...")
ratings_df = pd.read_csv('goodreads_ratings/goodreads_ratings.csv')  # user_id,book_id,rating columns

# Create user and item mappings to handle non-sequential IDs
print("Creating ID mappings...")
user_id_map = {id: i for i, id in enumerate(ratings_df['user_id'].unique())}
book_id_map = {id: i for i, id in enumerate(ratings_df['book_id'].unique())}

# Convert ratings to sparse matrix format
print("Converting to sparse matrix...")
row = ratings_df['user_id'].map(user_id_map)
col = ratings_df['book_id'].map(book_id_map)
data = ratings_df['rating'].astype(np.float32)

# Create sparse matrix (users x books)
sparse_ratings = csr_matrix(
    (data, (row, col)),
    shape=(len(user_id_map), len(book_id_map))
)

# Initialize ALS model
print("Training model...")
model = AlternatingLeastSquares(
    factors=50,  # number of latent factors
    iterations=20,
    regularization=0.1,
    random_state=42
)

# Fit the model
model.fit(sparse_ratings)

# Save the model and mappings
print("Saving model...")
import joblib
model_data = {
    'model': model,
    'user_id_map': user_id_map,
    'book_id_map': book_id_map,
    'inverse_user_map': {v: k for k, v in user_id_map.items()},
    'inverse_book_map': {v: k for k, v in book_id_map.items()}
}

joblib.dump(model_data, 'collaborative_model.joblib',
           compress=('gzip', 3),
           protocol=4)

print("Done!")