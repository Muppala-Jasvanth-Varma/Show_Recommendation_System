import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

data = pd.read_csv(r"W:\FOML\project\imdb_Top_250_TV_Shows.csv")

data['Episodes'] = data['Episodes'].str.replace('eps', '').astype(int)

data.rename(columns={'Rating given by people': 'votes'}, inplace=True)

# convert votes from string to numerical format
def convert_votes(votes):
    votes = votes.replace('(', '').replace(')', '')
    if 'M' in votes:
        return int(float(votes.replace('M', '')) * 1000000)
    elif 'K' in votes:
        return int(float(votes.replace('K', '')) * 1000)
    elif 'B' in votes:
        return int(float(votes.replace('B', '')) * 1000000000)
    else:
        return int(votes)


data['votes'] = data['votes'].apply(convert_votes)

# Rename the 'Release Year' column to 'year'
data.rename(columns={'Release Year': 'year'}, inplace=True)

# extract start year from year
def start_year(year):
    if '–' in year:
        return year.split('–')[0].strip()  # Extract start year
    else:
        return year.strip()  # Return year as it is

# extract end year from year
def end_year(year):
    if '–' in year:
        try:
            return year.split('–')[1].strip()  # Extract end year
        except IndexError:
            return None  # Return None if there's no valid end year
    else:
        return year.strip()  # Return the same year if there's no range

# extract 'start_year' and 'end_year'
data['start_year'] = data['year'].apply(start_year)

# Handle cases where 'start_year' or 'end_year' might be empty strings or invalid
data['start_year'] = pd.to_numeric(data['start_year'], errors='coerce') 

# Extract 'end_year' and convert to numeric, invalid values will be handled
data['end_year'] = data['year'].apply(end_year)
data['end_year'] = pd.to_numeric(data['end_year'], errors='coerce')  

# Fill missing 'end_year' values with 'start_year' values
data['end_year'] = data['end_year'].fillna(data['start_year'])

# Drop the original 'year' column
data.drop(columns=['year'], inplace=True)

# Calculate the 'duration' between start_year and end_year
data['duration'] = data['end_year'] - data['start_year']

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
data[['votes', 'duration']] = scaler.fit_transform(data[['votes', 'duration']])

# Create the feature matrix (using Rating, normalized Votes, and Duration)
features = data[['Rating', 'votes', 'duration']]

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(features)

# Function to recommend TV shows based on the selected show
def recommend_shows(show_name, top_n=5):
    # Get the index of the selected show
    idx = data[data['Shows Name'] == show_name].index[0]
    
    # Get similarity scores for the selected show
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort shows based on similarity score (descending order)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N most similar shows
    recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]  # Use top_n here
    
    # Return the recommended shows
    return data.iloc[recommended_indices][['Shows Name', 'Rating', 'votes', 'start_year', 'end_year']]

# Route for the recommendation API
@app.route('/recommend', methods=['GET'])
def recommend():
    tv_show_name = request.args.get('tv_show')
    if not tv_show_name:
        return jsonify({'error': 'Please provide a TV show name'}), 400
    
    recommended_shows = recommend_shows(tv_show_name, top_n=5)
    return jsonify(recommended_shows.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
