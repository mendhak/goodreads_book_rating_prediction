from textual import on
from textual.app import App, ComposeResult
from textual.validation import Function, Number, ValidationResult, Validator
from textual.widgets import Input, Label, Pretty, Button, Log
from textual.containers import Horizontal

import requests 
from bs4 import BeautifulSoup
import requests_cache
import json
requests_cache.install_cache('demo_cache')

import joblib
import pandas as pd

class InputApp(App):
    
    CSS = """
    Input.-valid {
        border: tall $success 60%;
    }
    Input.-valid:focus {
        border: tall $success;
    }
    Input {
        margin: 1 1;
    }
    Label {
        margin: 1 2;
    }
    Button {
        margin: 1 1;
    }
    Pretty {
        margin: 1 2;
    }
    Log {
        margin: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Enter a Goodreads Book URL:")
        yield Input(
            id="book_url",
            placeholder="Enter a Goodreads book URL",
            
        )
        with Horizontal():
    
            yield Button("Predict", id="get_info", variant="primary")
            yield Button("Exit", id="exit", variant="error")
        yield Log()
        

    def xyz(self):
        return "HELLO"

    def predict_book_rating(self, title, author, year_published, num_pages, genres):
        # Load the saved model and encoders
        model_data = joblib.load('goodreads_model_2.joblib')
        random_forest_model = model_data['model']
        author_encoder = model_data['author_encoder']
        mlb = model_data['genre_encoder']
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one(Log)
        if event.button.id == "exit":
            self.exit()
            return
        
        book_url = self.query_one("#book_url").value
        log.write_line(f"Book URL: {book_url}")

        response = requests.get(book_url)

        soup = BeautifulSoup(response.content, 'html.parser')

        genres = []
        author_name = 'Unknown Author'
        number_of_pages = 0
        book_name = ''
        publication_date = '2000'

        publication_date = soup.find("p", {"data-testid": "publicationInfo"}).text
        if publication_date:
            publication_date = publication_date.split(' ')[-1]

        for item in soup.select('a.Button--tag'):
            genre_url = item.attrs['href']
            if genre_url.startswith('https://www.goodreads.com/genres/'):
                genre = genre_url.split('/')[-1]
                genre_tag = genre_url.split('/genres/')[1]
                genres.append(genre_tag)

        # Get ld+json data
        ld_json_element = soup.find('script', type='application/ld+json')
        if ld_json_element:
            ld_json_data = json.loads(ld_json_element.string)
            number_of_pages = ld_json_data['numberOfPages']
            author_name = ld_json_data['author'][0]['name']
            book_name = ld_json_data['name']

        # log.write_line(f"""
        # Book Name: {book_name}
        # Author: {author_name}
        # Publication Date: {publication_date}
        # Number of Pages: {number_of_pages}
        # Genres: {', '.join(genres)}""")


        prediction = self.predict_book_rating(title=book_name,
                    author=author_name,
                    year_published=publication_date,
                    num_pages=number_of_pages,
                    genres=genres
                    )
        log.write_line(f"\n{prediction['Title']}:")
        log.write_line(f"Predicted Rating: {prediction['Predicted Rating']:.2f}")
        log.write_line(f"Recommendation: {prediction['Rating Sentence']}")
        log.write_line(f"Genres used: {prediction['Used Genres']}")  
                       



    


app = InputApp()

if __name__ == "__main__":
    app.run()