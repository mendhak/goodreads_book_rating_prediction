uv sync

## Generating the models

For the content model, based on my own goodreads data:  

uv run generate_content_model.py

For the collaborative model, based on the goodreads dataset:

```
wget https://datascience.quantecon.org/assets/data/goodreads_ratings.csv.zip
unzip goodreads_ratings.csv.zip -d goodreads_ratings/
# misnamed for some reason
mv goodreads_ratings/goodreads_ratings.csv.zip goodreads_ratings/goodreads_ratings.csv
```

uv run generate_collaborative_model.py


## Running the predictors

uv run textual_goodreads_predictor.py

or

uv run predict_rating.py



## TODO?

Could I add weight to genres.

def create_genre_weights(df):
    genre_counts = df['Bookshelves'].value_counts()
    return {genre: 1 + np.log(count) for genre, count in genre_counts.items()}


Could I include average ratings, number of ratings, number of reviews. 

