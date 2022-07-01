# utils import
from torch.utils.data import Dataset
from torch import as_tensor, Tensor, float
class IDMDHDataset(Dataset):
    '''
    simple class to load pandas DataFrame as Datasets in pytorch
    '''
    def __init__(self, df, features, weights = "luminosity_weight") -> None:
        super().__init__()
        self.df = df
        self.train_labels = features
        self.weights = weights
        self.train_data = Tensor(self.df.loc[self.df.index,self.train_labels].values)
        #self.weight_data = Tensor(self.df.loc[self.df.index,self.weights].values)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tensor:
        return self.train_data[index], 0