import numpy as np
import itertools
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

# Similarity
def get_similarity(a, b):
    va = v[a]
    vb = v[b]
    return (va.T @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb))

# similarities = []
# for (a, b) in itertools.combinations(movies_dict.keys(), 2):
#     va = v[a]
#     vb = v[b]
#     similarity = (va.T @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
#     similarities.append(({a, b}, similarity))

# def get_most_similar(id, count):
#     return [list(pair - {id})[0] for (pair, similarity) in sorted([(pair, similarity) for (pair, similarity) in similarities if id in pair], key=lambda row: row[1], reverse=True)[:count]]

# print(get_most_similar(260, 10))
# print(get_most_similar(2628, 10))
# np.mean([similarity for (_pair, similarity) in similarities])

# similarity_by_genre = {}
# for genre in genres:
#     ids = [id for id, item_genres in genres_dict.items() if genre in item_genres]
#     similarities = []
#     for a in ids:
#         for b in ids:
#             if a == b: continue
#             similarities.append(get_similarity(a, b))
#     similarity_by_genre[genre] = np.mean(similarities)

# print(sorted(similarity_by_genre.items(), key=lambda row: row[1])[-1])
# print(sorted(similarity_by_genre.items(), key=lambda row: row[1])[0])

genre_to_ids = {genre: [] for genre in genres}
for id, item_genres in genres_dict.items():
    for genre in item_genres:
        genre_to_ids[genre] += [id]

comedy_similarities = {}
for genre in genres:
    similarities = []
    for a in genre_to_ids["Comedy"]:
        for b in genre_to_ids[genre]:
            if a == b: continue
            similarities.append(get_similarity(a, b))
    comedy_similarities[genre] = np.mean(similarities)
