import pandas as pd 
# Taken from the .ipynb only the relevant parts. 

df = pd.read_csv('goodreads_library_export.csv')
read_books_df = df[df['Exclusive Shelf'] == 'read']

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

mlb = MultiLabelBinarizer()
book_genres = read_books_df['Bookshelves'].fillna('').str.split(', ')
genre_features = pd.DataFrame(mlb.fit_transform(book_genres), columns=mlb.classes_, index=read_books_df.index)

# Create feature matrix and handle authors
features = pd.DataFrame(index=read_books_df.index)
author_encoder = LabelEncoder()
encoded_authors = author_encoder.fit_transform(read_books_df['Author'].fillna('Unknown Author'))
features['Author'] = encoded_authors

features['Year Published'] = read_books_df['Year Published']
features['Number of Pages'] = read_books_df['Number of Pages']

features = pd.concat([features, genre_features], axis=1)

features = features.fillna({
    'Number of Pages': 0,
    'Year Published': 2000
})


training_X = features.copy()  # This is the feature set, the 'input' so to speak. Could add more kinds of features later like sentiment?
training_Y = read_books_df['My Rating']  # this is the target to predict

# Remove where rating = 0, or only include where books have ratings
training_X = training_X[training_Y > 0]
training_Y = training_Y[training_Y > 0]


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train, X_test, Y_train, Y_text = train_test_split(training_X, training_Y, test_size=0.2, random_state=23498234)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=128383)

random_forest_model.fit(X_train, Y_train)

Y_predicted = random_forest_model.predict(X_test)

mse = mean_squared_error(Y_text, Y_predicted)
mae = mean_absolute_error(Y_text, Y_predicted)

to_read = df[df['Exclusive Shelf'] == 'to-read'].copy()
to_read.loc[:, 'Bookshelves'] = to_read['Bookshelves'].str.replace('to-read, ', '').str.replace('to-read', '')

to_read_features = pd.DataFrame(index=to_read.index)

all_authors = pd.concat([read_books_df['Author'], to_read['Author']]).unique()
author_encoder.fit(all_authors)
to_read_features['Author'] = author_encoder.transform(to_read['Author'].fillna('Unknown Author'))

to_read_features['Year Published'] = to_read['Year Published']
to_read_features['Number of Pages'] = to_read['Number of Pages']

# Get genres - make sure to use the same index
unread_genres = to_read['Bookshelves'].fillna('').str.split(', ')
unread_genre_features = pd.DataFrame(
    mlb.transform(unread_genres), 
    columns=mlb.classes_,
    index=to_read.index
)

to_read_features = pd.concat([to_read_features, unread_genre_features], axis=1)

# Fill missing values
to_read_features = to_read_features.fillna({
    'Number of Pages': 0,
    'Year Published': 2000
})

# print("Shapes after processing:")
# print("to_read shape:", to_read.shape)
# print("to_read_features shape:", to_read_features.shape)

# Make predictions
predicted_ratings = random_forest_model.predict(to_read_features)

import joblib

model_data = {
    'model': random_forest_model,
    'author_encoder': author_encoder,
    'genre_encoder': mlb
}

# Save with highest protocol and compressed numpy arrays
joblib.dump(model_data, 'goodreads_model_2.joblib', 
           compress=('gzip', 3), 
           protocol=4)