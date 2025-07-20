import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    # Replace this with your actual data loading
    data = {
        'user_id': [1, 1, 2, 2, 3, 3],
        'food_id': [101, 102, 101, 103, 102, 104],
        'rating': [5, 3, 4, 2, 5, 4]
    }
    df = pd.DataFrame(data)
    matrix = df.pivot_table(index='user_id', columns='food_id', values='rating').fillna(0)
    return matrix

def get_recommendations(user_id, matrix):
    try:
        user_id = int(user_id)
    except:
        return []

    if user_id not in matrix.index:
        return []

    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    similarity = cosine_similarity(user_vector, matrix)[0]
    
    sim_scores = pd.Series(similarity, index=matrix.index)
    sim_scores = sim_scores.drop(user_id)  # exclude the user themself
    most_similar_user = sim_scores.idxmax()

    similar_user_ratings = matrix.loc[most_similar_user]
    target_user_ratings = matrix.loc[user_id]

    # Recommend food the similar user liked that the target user hasn't rated
    recommendations = similar_user_ratings[target_user_ratings == 0].sort_values(ascending=False)
    
    return list(recommendations.index)



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    matrix = load_data()
    recs = get_recommendations(user_id, matrix)
    
    if not recs:
        return jsonify({"error": "User not found"}), 404
    return jsonify(recs)

# ✅ THIS is what makes the Flask app run when you execute `python app.py`
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
