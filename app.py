from flask import Flask, render_template, request
import pandas as pd
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import process

nltk.download('vader_lexicon')
nltk.download('punkt')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# Load the dataset
movies_df = pd.read_csv("processed_movies_full.csv")

# Clean and format cast
def get_cast_names(cast):
    try:
        cast_list = eval(cast)
        return ', '.join([member['name'] for member in cast_list[:3]])
    except:
        return ''
movies_df['cast'] = movies_df['cast'].apply(get_cast_names)

# Format genres
movies_df['genre'] = movies_df['genre'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])

# Poster and backdrop URL generation
image_base_url = 'https://image.tmdb.org/t/p/w500'
movies_df['poster_url'] = movies_df['poster_path'].apply(lambda x: image_base_url + x if isinstance(x, str) and x.startswith('/') else '')
movies_df['backdrop_url'] = movies_df['backdrop_path'].apply(lambda x: image_base_url + x if isinstance(x, str) and x.startswith('/') else '')

# Expanded mood-to-genre mapping
mood_genre_map = {
    # Comedy
    "happy": ['Comedy'], "funny": ['Comedy'], "i want to laugh": ['Comedy'],
    "laugh out loud": ['Comedy'], "hilarious": ['Comedy'], "stand up": ['Comedy'],
    "joke": ['Comedy'], "cheerful": ['Comedy'], "giggle": ['Comedy'],
    "fun night": ['Comedy'], "i want comedy": ['Comedy'],

    # Sadness
    "sad": ['Drama'], "down": ['Drama'], "depressed": ['Drama'],
    "cry": ['Drama'], "tearjerker": ['Drama'], "low": ['Drama'],
    "i want to cry": ['Drama'], "emotional": ['Drama'], "melancholy": ['Drama'],

    # Romance
    "romantic": ['Romance'], "in love": ['Romance'], "love story": ['Romance'],
    "heartbroken": ['Romance', 'Drama'], "breakup": ['Romance'],
    "date night": ['Romance'], "valentine": ['Romance'], "relationship": ['Romance'],

    # Action
    "excited": ['Action'], "thrill ride": ['Action'], "intense": ['Action'],
    "fight": ['Action'], "war": ['Action'], "battle": ['Action'],
    "superhero": ['Action'], "adrenaline": ['Action'], "high energy": ['Action'],

    # Horror
    "horror": ['Horror'], "scary": ['Horror'], "ghost": ['Horror'],
    "haunted": ['Horror'], "paranormal": ['Horror'], "jump scare": ['Horror'],
    "creepy": ['Horror'], "frightening": ['Horror'], "zombies": ['Horror'],

    # Thriller
    "thrill": ['Thriller'], "mystery": ['Mystery'], "twist": ['Thriller'],
    "suspense": ['Thriller'], "tense": ['Thriller'], "crime": ['Crime'],
    "detective": ['Mystery'], "investigation": ['Mystery'], "heist": ['Thriller'],

    # Science Fiction
    "sci fi": ['Science Fiction'], "space": ['Science Fiction'],
    "future": ['Science Fiction'], "robots": ['Science Fiction'],
    "aliens": ['Science Fiction'], "cyberpunk": ['Science Fiction'],
    "technology": ['Science Fiction'], "dystopia": ['Science Fiction'],

    # Fantasy
    "magic": ['Fantasy'], "fantasy": ['Fantasy'], "fairy tale": ['Fantasy'],
    "dragon": ['Fantasy'], "wizard": ['Fantasy'], "medieval": ['Fantasy'],
    "myth": ['Fantasy'],

    # Adventure
    "bored": ['Adventure'], "adventure": ['Adventure'],
    "treasure hunt": ['Adventure'], "exploration": ['Adventure'],
    "journey": ['Adventure'], "wild": ['Adventure'], "epic": ['Adventure'],
     "low": ['Drama'],
"i feel low": ['Drama'],
"feeling low": ['Drama'],
"my mood is low": ['Drama'],
"i am feeling low": ['Drama'],
"emotionally low": ['Drama'],

    # Animation & Family
    "kids": ['Animation', 'Family'], "family": ['Family'],
    "disney": ['Animation'], "pixar": ['Animation'], "animated": ['Animation'],
    "cartoon": ['Animation'], "light hearted": ['Animation'],

    # Documentary / Real
    "true story": ['Biography'], "real": ['Biography'],
    "based on true events": ['Biography'], "docu": ['Documentary'],
    "educational": ['Documentary'], "learn": ['Documentary'],

    # Musical / Dance
    "music": ['Music'], "musical": ['Music'], "concert": ['Music'],
    "dance": ['Music'], "performance": ['Music'],

    # Historical / Biographical
    "history": ['History'], "biography": ['Biography'],
    "historical drama": ['History', 'Drama'], "past": ['History'],

    # Western / Classic / Noir
    "western": ['Western'], "cowboy": ['Western'],
    "black and white": ['Mystery'], "classic": ['Drama'],

    # Sports
    "sports": ['Sport'], "game": ['Sport'], "competition": ['Sport'],
    "football": ['Sport'], "basketball": ['Sport'], "athlete": ['Sport']
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', movies=[], query='')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input_raw = request.form['mood']
    user_input = re.sub(r'[^\w\s]', '', user_input_raw.lower().strip())

    # Step 1: Exact match
    matched_genres = mood_genre_map.get(user_input, [])

    # Step 2: Substring match
    if not matched_genres:
        for phrase, genres in mood_genre_map.items():
            if phrase in user_input:
                matched_genres.extend(genres)

    # Step 3: Fuzzy fallback
    if not matched_genres:
        all_genres = set(genre.lower() for sublist in movies_df['genre'] for genre in sublist)
        best_match, _ = process.extractOne(user_input, list(all_genres))
        matched_genres = [best_match.capitalize()]

    # Step 4: Filter movies
    matched_movies = movies_df[movies_df['genre'].apply(
        lambda genre_list: any(g in genre_list for g in matched_genres)
    )].drop_duplicates(subset='title')

    if matched_movies.empty:
        matched_movies = movies_df.drop_duplicates(subset='title').sample(5)

    # Step 5: Format
    movies = []
    for _, row in matched_movies.iterrows():
        movies.append({
            'title': row['title'],
            'genre': row['genre'],
            'director': row.get('director', ''),
            'cast': row['cast'],
            'overview': row['overview'],
            'trailer': row.get('trailer', '#'),
            'poster_url': row.get('poster_url', ''),
            'backdrop_url': row.get('backdrop_url', '')
        })

    return render_template('index.html', movies=movies, query=user_input_raw)

if __name__ == '__main__':
    app.run(debug=True)