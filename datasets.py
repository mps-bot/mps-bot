from typing import Dict, Tuple
import torch
import os
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.loader import NeighborLoader
from utils import sample_mask

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MGTAB_PATH = os.path.join(CURRENT_DIR, "Dataset", "MGTAB")
TWIBOT22_PATH = os.path.join(CURRENT_DIR, "Dataset", "TwiBot22")


# MGTAB
class MGTAB(InMemoryDataset):
    def __init__(self, root=MGTAB_PATH, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['edge_index.pt', 'edge_type.pt', 'edge_weight.pt',
                'labels_stance.pt', 'labels_bot.pt', 'features.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        edge_index = torch.load(self.root + "/edge_index.pt")
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_type = torch.load(self.root + "/edge_type.pt")
        edge_weight = torch.load(self.root + "/edge_weight.pt")
        stance_label = torch.load(self.root + "/labels_stance.pt")
        bot_label = torch.load(self.root + "/labels_bot.pt")

        features = torch.load(self.root + "/features.pt").to(torch.float32)

        data = Data(x=features, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.y1 = stance_label
        data.y2 = bot_label

        sample_number = len(data.y1)
        train_idx = range(int(0.7 * sample_number))
        val_idx   = range(int(0.7 * sample_number), int(0.9 * sample_number))
        test_idx  = range(int(0.9 * sample_number), int(sample_number))

        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask   = sample_mask(val_idx, sample_number)
        data.test_mask  = sample_mask(test_idx, sample_number)

        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_mgtab() -> Dict[str, torch.Tensor]:
    dataset = MGTAB(MGTAB_PATH)
    data = dataset[0]
    return dict(
        x=data.x,
        y=data.y2.long(),
        edge_index=data.edge_index,
        edge_type=data.edge_type.long(),
        train_mask=data.train_mask.bool(),
        val_mask=data.val_mask.bool(),
        test_mask=data.test_mask.bool(),
    )

# TwiBot-22
def load_twibot22():
    x = torch.cat([
        torch.load(TWIBOT22_PATH + "/cat_properties_tensor.pt",  map_location="cpu"),
        torch.load(TWIBOT22_PATH + "/num_properties_tensor.pt",  map_location="cpu"),
        torch.load(TWIBOT22_PATH + "/tweets_tensor.pt",          map_location="cpu"),
        torch.load(TWIBOT22_PATH + "/des_tensor.pt",             map_location="cpu")
    ], dim=1)
    edge_index = torch.load(TWIBOT22_PATH + "/edge_index.pt", map_location="cpu")
    edge_type  = torch.load(TWIBOT22_PATH + "/edge_type.pt",  map_location="cpu")
    label      = torch.load(TWIBOT22_PATH + "/label.pt",      map_location="cpu").long()

    data = HeteroData()
    data["user"].x = x
    data["user"].y = label
    data["user", "follower", "user"].edge_index  = edge_index[:, edge_type == 0]
    data["user", "following", "user"].edge_index = edge_index[:, edge_type == 1]

    data.train_idx = torch.load(TWIBOT22_PATH + "/train_idx.pt", map_location="cpu").long()
    data.valid_idx = torch.load(TWIBOT22_PATH + "/val_idx.pt",   map_location="cpu").long()
    data.test_idx  = torch.load(TWIBOT22_PATH + "/test_idx.pt",  map_location="cpu").long()
    return data


def make_twibot22_loaders(data: HeteroData, batch_size=256, test_batch_size=200, num_neighbors=256) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
    train_loader = NeighborLoader(
        data,
        input_nodes=("user", data.train_idx),
        num_neighbors=[num_neighbors, num_neighbors],
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=("user", data.valid_idx),
        num_neighbors=[num_neighbors, num_neighbors],
        batch_size=batch_size
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=("user", data.test_idx),
        num_neighbors=[num_neighbors, num_neighbors],
        batch_size=test_batch_size
    )
    return train_loader, val_loader, test_loader
