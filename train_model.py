import pandas as pd
import torch
from torch.utils.data import DataLoader
from recommender import NCF, MovieDataset

# Load and preprocess data
ratings = pd.read_csv("/home/parth/Downloads/ml-latest-small/ratings.csv")

# Normalize ratings to 0–1 range
ratings['rating'] = ratings['rating'] / 5.0

# Create mappings
user_mapping = {id: idx for idx, id in enumerate(ratings['userId'].unique())}
movie_mapping = {id: idx for idx, id in enumerate(ratings['movieId'].unique())}

ratings['user'] = ratings['userId'].map(user_mapping)
ratings['movie'] = ratings['movieId'].map(movie_mapping)

# Dataset
dataset = MovieDataset(ratings['user'], ratings['movie'], ratings['rating'])
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Model
model = NCF(num_users=len(user_mapping), num_items=len(movie_mapping))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):  # Increase for better accuracy
    model.train()
    running_loss = 0.0
    for user, movie, rating in dataloader:
        user = user.to(device)
        movie = movie.to(device)
        rating = rating.to(device)

        optimizer.zero_grad()
        output = model(user, movie)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), '/home/parth/Downloads/movies/NCF_Recommender/best_model.pth')
print("✅ Model saved successfully.")
