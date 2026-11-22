import pandas as pd
import ast

# Load datasets
movies = pd.read_csv("Movies 10k.csv")
credits = pd.read_csv("Credit 10k.csv")

# Merge datasets on 'id'
df = movies.merge(credits, on="id")

# Keep necessary columns
df = df[['title', 'overview', 'genre', 'cast', 'crew']]

# Extract top 3 cast
def extract_cast(cast_str):
    try:
        cast = ast.literal_eval(cast_str)
        return ", ".join([c['name'] for c in cast[:3]])
    except:
        return ""

# Extract director from crew
def extract_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for c in crew:
            if c['job'] == 'Director':
                return c['name']
        return ""
    except:
        return ""

df['top_cast'] = df['cast'].apply(extract_cast)
df['director'] = df['crew'].apply(extract_director)

# Drop original cast and crew
df.drop(['cast', 'crew'], axis=1, inplace=True)

# Mood mapping by genre keywords
genre_to_mood = {
    "comedy": "happy",
    "animation": "happy",
    "family": "happy",
    "drama": "sad",
    "romance": "romantic",
    "thriller": "excited",
    "horror": "scared",
    "action": "energetic",
    "adventure": "curious",
    "fantasy": "dreamy",
    "crime": "serious",
    "mystery": "intrigued",
    "science fiction": "curious"
}

def infer_mood(genre_str):
    try:
        genre_str = genre_str.lower()
        for g in genre_to_mood:
            if g in genre_str:
                return genre_to_mood[g]
        return "neutral"
    except:
        return "neutral"

df['mood'] = df['genre'].apply(infer_mood)

# Save processed CSV
df.to_csv("processed_movies.csv", index=False)
print("✅ Processed dataset saved as 'processed_movies.csv'")