import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import gc

# Dataset class
class MovieDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.movie_ids[idx], self.ratings[idx])

# NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, layers=[128, 64, 32]):
        super(NCF, self).__init__()
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.ModuleList()
        input_dim = 2 * embedding_dim
        for layer_size in layers:
            self.mlp.append(nn.Linear(input_dim, layer_size))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.BatchNorm1d(layer_size))
            self.mlp.append(nn.Dropout(p=0.3))
            input_dim = layer_size

        self.output = nn.Linear(layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, user_indices, item_indices):
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = user_embedding_gmf * item_embedding_gmf

        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

        for layer in self.mlp:
            mlp_vector = layer(mlp_vector)

        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        rating = self.sigmoid(self.output(vector))
        return rating.squeeze()

# Recommender class
class MovieRecommender:
    def __init__(self, data, embedding_dim=16, layers=[128, 64, 32]):
        self.data = data
        self.embedding_dim = embedding_dim
        self.layers = layers

        self.movie_info = data[['movieId', 'title', 'genres']].drop_duplicates().set_index('movieId')
        self.user_mapping = {id: idx for idx, id in enumerate(data['userId'].unique())}
        self.movie_mapping = {id: idx for idx, id in enumerate(data['movieId'].unique())}
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_movie_mapping = {v: k for k, v in self.movie_mapping.items()}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NCF(
            num_users=len(self.user_mapping),
            num_items=len(self.movie_mapping),
            embedding_dim=embedding_dim,
            layers=layers
        ).to(self.device)

        self.model.load_state_dict(torch.load('/home/parth/Downloads/movies/NCF_Recommender/best_model.pth', map_location=self.device))
        self.model.eval()

    def get_recommendations(self, user_id, n_recommendations=10):
        if user_id not in self.user_mapping:
            return []

        user_idx = self.user_mapping[user_id]
        user_tensor = torch.tensor([user_idx] * len(self.movie_mapping), device=self.device)
        movie_tensor = torch.tensor(list(range(len(self.movie_mapping))), device=self.device)

        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor)

        recommendations = []
        movie_scores = [(self.reverse_movie_mapping[idx], score.item() * 5.0)
                        for idx, score in enumerate(predictions)]

        for movie_id, pred_rating in sorted(movie_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]:
            title = self.movie_info.loc[movie_id, 'title']
            genres = self.movie_info.loc[movie_id, 'genres']
            recommendations.append({
                'movie_id': movie_id,
                'title': title,
                'genres': genres,
                'predicted_rating': round(pred_rating, 2)
            })

        return recommendations

# Recommender factory
def create_recommender(data):
    return MovieRecommender(data)

# Debugging & test
if __name__ == "__main__":
    ratings = pd.read_csv("/home/parth/Downloads/ml-latest-small/ratings.csv")
    movies = pd.read_csv("/home/parth/Downloads/ml-latest-small/movies.csv")
    merged_data = pd.merge(ratings, movies, on='movieId')
    merged_data['genres'] = merged_data['genres'].apply(lambda x: ' '.join(sorted(x.split('|'))))

    recommender = MovieRecommender(merged_data)
    print("✅ Recommender initialized")
    print("Available methods:", dir(recommender))
    # Optional test
    user_id = 1
    recs = recommender.get_recommendations(user_id=user_id)
    for r in recs:
        print(f"{r['title']} ({r['genres']}) - Predicted Rating: {r['predicted_rating']}")
