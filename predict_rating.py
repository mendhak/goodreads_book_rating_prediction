import joblib
import pandas as pd

# Load the saved model and encoders
model_data = joblib.load('goodreads_model.joblib')
random_forest_model = model_data['model']
author_encoder = model_data['author_encoder']
mlb = model_data['genre_encoder']

def predict_book_rating(title, author, year_published, num_pages, genres):
    
    book_features = pd.DataFrame(index=[0])
    
    # Handle unknown authors
    try:
        book_features['Author'] = author_encoder.transform([author])
    except ValueError:
        book_features['Author'] = author_encoder.transform(['Unknown Author'])
    
    book_features['Year Published'] = year_published
    book_features['Number of Pages'] = num_pages
    
    # Handle genres - only use known genres
    known_genres = list(set(genres) & set(mlb.classes_))
    if not known_genres:
        known_genres = ['']
        
    genre_features = pd.DataFrame(
        mlb.transform([known_genres]), 
        columns=mlb.classes_,
        index=[0]
    )
    
    book_features = pd.concat([book_features, genre_features], axis=1)
    predicted_rating = random_forest_model.predict(book_features)[0]
    
    return {
        'Title': title,
        'Predicted Rating': predicted_rating,
        'Rating Sentence': "You might like it" if predicted_rating >= 2.8 
                         else ("It might be okay" if predicted_rating >= 2.0 
                         else "You probably won't like it"),
        'Used Genres': known_genres
    }

# Example usage
test_book = {
    'title': 'Project Hail Mary',
    'author': 'Andy Weir',
    'year': 2021,
    'pages': 496,
    'genres': ['science-fiction', 'fiction', 'dystopian']
}

result = predict_book_rating(
    test_book['title'],
    test_book['author'],
    test_book['year'],
    test_book['pages'],
    test_book['genres']
)

print(f"\n{result['Title']}:")
print(f"Predicted Rating: {result['Predicted Rating']:.2f}")
print(f"Recommendation: {result['Rating Sentence']}")
print(f"Genres used: {result['Used Genres']}")