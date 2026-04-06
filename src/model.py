import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index):

        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # On fusionne les deux types de nœuds pour le calcul du graphe
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Liste pour stocker les sorties de chaque couche (pour la moyenne finale)
        out = x / (len(self.convs) + 1)
        
        # 3. Propagation du message (les voisins s'influencent)
        for conv in self.convs:
            x = conv(x, edge_index)
            out = out + x / (len(self.convs) + 1)
            
        # On sépare à nouveau les utilisateurs et les films
        users, items = torch.split(out, [self.user_embedding.num_embeddings, self.item_embedding.num_embeddings])
        return users, items