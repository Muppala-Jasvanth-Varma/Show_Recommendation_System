# TV Show Recommendation System

## Overview
This project implements a TV show recommendation system using a content-based filtering approach with cosine similarity. The system is built using Python, pandas, scikit-learn, and Flask to provide an API endpoint for recommendations.

## Dataset
The system uses the `imdb_Top_250_TV_Shows.csv` dataset, which includes:
- Show names
- Ratings
- Number of votes
- Release years (start and end year)
- Number of episodes

## Data Preprocessing
1. **Cleaning and Formatting**:
   - The `Episodes` column is cleaned by removing 'eps' and converting it to an integer.
   - The `Rating given by people` column is renamed to `votes` and converted into a numerical format.
   - Votes are scaled based on 'K' (thousands), 'M' (millions), and 'B' (billions).
   - The `Release Year` column is split into `start_year` and `end_year`.
   - Missing `end_year` values are filled with `start_year`.
   - The `duration` (difference between `start_year` and `end_year`) is calculated.

2. **Feature Scaling**:
   - The `votes` and `duration` columns are normalized using MinMaxScaler.

3. **Feature Matrix**:
   - The features used for recommendation include `Rating`, normalized `votes`, and `duration`.

## Similarity Calculation
- Cosine similarity is used to compute the similarity between TV shows based on the selected features.

## API Implementation
A Flask API is built with the following endpoint:
- **GET /recommend?tv_show=SHOW_NAME**: Returns the top 5 recommended TV shows based on the given show name.

## Running the Application
1. Ensure all dependencies are installed (`pandas`, `numpy`, `scikit-learn`, `Flask`).
2. Run the Flask app:
   ```sh
   python app.py
   ```
3. Access recommendations via:
   ```sh
   http://127.0.0.1:5000/recommend?tv_show=Breaking Bad
   ```

## Example API Response
```json
[
  {
    "Shows Name": "Better Call Saul",
    "Rating": 8.7,
    "votes": 0.85,
    "start_year": 2015,
    "end_year": 2022
  },
  {
    "Shows Name": "The Wire",
    "Rating": 9.3,
    "votes": 0.75,
    "start_year": 2002,
    "end_year": 2008
  }
]
```

## Future Improvements
- Include more features like genres, actors, and directors for better recommendations.
- Implement collaborative filtering for improved accuracy.
- Deploy the model as a cloud-based API service.
