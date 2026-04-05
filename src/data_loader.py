import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

def load_movielens_graph(ratings_path, movies_path):
    # Chargement des données
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    # Mapping des identifiants vers des index 0 à N
    unique_user_id = ratings_df['userId'].unique()
    user_mapping = {id: i for i, id in enumerate(unique_user_id)}
    unique_movie_id = ratings_df['movieId'].unique()
    movie_mapping = {id: i for i, id in enumerate(unique_movie_id)}

    # Initialisation de l'objet HeteroData
    data = HeteroData()
    data['user'].num_nodes = len(unique_user_id)
    data['movie'].num_nodes = len(unique_movie_id)

    # Filtrage des interactions positives (notes >= 3.0)
    mask = ratings_df['rating'] >= 3.0
    edge_index_user = ratings_df[mask]['userId'].map(user_mapping).values
    edge_index_movie = ratings_df[mask]['movieId'].map(movie_mapping).values

    # Construction des arêtes (User -> Rates -> Movie)
    data['user', 'rates', 'movie'].edge_index = torch.stack([
        torch.tensor(edge_index_user, dtype=torch.long),
        torch.tensor(edge_index_movie, dtype=torch.long)
    ], dim=0)

    # Ajout des arêtes inverses pour le Message Passing bidirectionnel
    data = T.ToUndirected()(data)

    # Définition du split Train/Val/Test
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=('user', 'rates', 'movie'),
        rev_edge_types=('movie', 'rev_rates', 'user'),
    )

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data