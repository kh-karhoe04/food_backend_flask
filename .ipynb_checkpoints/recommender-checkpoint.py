import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv('ratings.csv')
    matrix = df.pivot_table(index='user_id', columns='food', values='rating').fillna(0)
    return matrix

def get_user_similarity(matrix):
    return cosine_similarity(matrix)

def get_recommendations(user_id, matrix):
    if user_id not in matrix.index:
        return {}
    
    user_index = list(matrix.index).index(user_id)
    sim_matrix = get_user_similarity(matrix)
    sim_scores = sim_matrix[user_index]
    
    sim_df = pd.Series(sim_scores, index=matrix.index).drop(user_id)
    weighted_ratings = matrix.T.dot(sim_df) / sim_df.sum()
    user_rated = matrix.loc[user_id]
    
    recommendations = weighted_ratings[user_rated == 0]
    return recommendations.sort_values(ascending=False).head(5).to_dict()
