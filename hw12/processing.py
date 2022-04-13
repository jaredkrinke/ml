import numpy as np
from code_for_hw12 import *

data = load_ratings_data()
movies_dict, genres_dict = load_movies()
model = load_model()

# Favorite genre
user_synthetic_id = 270894
genre_counts = {genre: 0 for genre in genres}
for i in [i for (a, i, r) in data if a == user_synthetic_id and r == 5]:
    for genre in genres_dict[i]:
        genre_counts[genre] += 1
print(max(genre_counts.items(), key=lambda x: x[1])[0])

# Best predictions
(u, b_u, v, b_v) = model
user_synthetic = u[user_synthetic_id]
user_synthetic_bias = b_u[user_synthetic_id]
# TODO: All this mixing of lists and NumPy arrays is tedious...
predictions = (user_synthetic.T @ np.hstack(v) + user_synthetic_bias + b_v).T[:,0]
user_synthetic_rated = {i for (a, i, _) in data if a == user_synthetic_id}
relevant_predictions = [(i, p) for (i, p) in enumerate(predictions) if not i in user_synthetic_rated and i in genres_dict]
relevant_predictions.sort(key=lambda a: a[1], reverse=True)
print(sum([1 if 'Animation' in genres_dict[i] else 0 for (i, _) in relevant_predictions[:50]]))
