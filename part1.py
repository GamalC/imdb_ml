import pandas as pd
from collections import Counter
from statistics import mean

imdb_path = "imdb_newdataset/imdb_database.csv"

"""
Part 1
1. What are the mean, minimum, and maximum movie scores?
2. What director filmed the most movies?
3. In how many different movies did Leonardo DiCaprio feature in? How many
of these had “Action” as one of their movie types?
"""

imdb_df = pd.read_csv(imdb_path)
imdb_df = imdb_df.drop_duplicates()

#Obtain necesary stats about movie scores
def get_movie_score_stats():
    #Read scores column in as list
    movie_scores = list(imdb_df['Score'])
    #Calculate various stats
    mean_score = mean(movie_scores)
    min_score = min(movie_scores)
    max_score = max(movie_scores)
    return round(mean_score, 3), min_score, max_score

#Obtain the director who has directed the most movies
def top_director():
    filtered_imdb_df = imdb_df.loc[imdb_df['Director'] != '[]'] #Remove mivies where director info is missing.
    directors = filtered_imdb_df['Director']
    directors_counter = Counter(directors)
    top_director = directors_counter.most_common(1) #Director with highest movie count
    return top_director[0][0].replace('[', '').replace(']', '')

#Get film count of a given actor, optionally with particular movie type
def get_actor_filmcount(actor_name, movie_type=''):
    cnt = 0
    #TODO: more efficient method that iterrows() which is slow.
    for _, row in imdb_df.iterrows():
        if actor_name in row['Actors'] and movie_type in row['Movie Type']:
            cnt += 1
    return cnt

if __name__ == '__main__':
    mean_score, min_score, max_score = get_movie_score_stats()
    print(f"Mean, Min, Max scores: {mean_score}, {min_score}, {max_score}")
    td_name = top_director()
    print(f"{td_name} directed the most movies.")
    leo_movies_cnt = get_actor_filmcount('Leonardo DiCaprio')
    leo_action_movies_cnt = get_actor_filmcount('Leonardo DiCaprio', movie_type='Action')
    print(f"Leonardo DiCaprio featured in {leo_movies_cnt} movies, {leo_action_movies_cnt} of them were action movies.")