import pandas as pd
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('punkt')

# Load datasets
movies = pd.read_csv('movies 10k.csv')
credits = pd.read_csv('credit 10k.csv')

# Rename and merge
credits.columns = ['id', 'cast', 'crew']
movies = movies.merge(credits[['id', 'cast', 'crew']], on='id', how='left')

# Prepare NLTK sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Preprocess and tag mood
def detect_mood(text):
    text = str(text).lower()
    score = sia.polarity_scores(text)['compound']
    
    if any(word in text for word in ['love', 'romance', 'crush']):
        return 'romantic'
    elif any(word in text for word in ['fear', 'kill', 'dark', 'ghost', 'zombie']):
        return 'horror'
    elif any(word in text for word in ['laugh', 'funny', 'hilarious']):
        return 'comedy'
    elif any(word in text for word in ['war', 'fight', 'battle', 'chase']):
        return 'action'
    elif any(word in text for word in ['magic', 'adventure', 'fantasy']):
        return 'adventure'
    elif score > 0.4:
        return 'joyful'
    elif score < -0.4:
        return 'sad'
    else:
        return 'neutral'

# Fill missing overviews with empty string
movies['overview'] = movies['overview'].fillna('')
movies['mood'] = movies['overview'].apply(detect_mood)

# Extract director name
def get_director(crew_str):
    try:
        crew = eval(crew_str)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
    except:
        return None

movies['director'] = movies['crew'].apply(get_director)

# Extract top 3 cast names
def get_top_cast(cast_str):
    try:
        cast = eval(cast_str)
        return ', '.join([member.get('name') for member in cast[:3]])
    except:
        return ''

movies['top_cast'] = movies['cast'].apply(get_top_cast)

# Add trailer search link (YouTube search URL)
def trailer_link(title):
    search_query = re.sub(r'[^\w\s]', '', title).replace(' ', '+') + "+trailer"
    return f"https://www.youtube.com/results?search_query={search_query}"

movies['trailer'] = movies['title'].apply(trailer_link)

# Save to new processed file
movies.to_csv('processed_movies_full.csv', index=False)
print("✅ Processed and saved with moods and trailer links.")