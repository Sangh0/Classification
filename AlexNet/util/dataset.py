from glob import glob
from PIL import Image

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(
        self, 
        path, 
        subset='train', 
        transforms_=None,
    ):
        assert subset in ('train', 'valid', 'test'), \
            'you should be choose between train, valid and test'
        
        dog_files = glob(path+'/'+subset+'/dogs/*.jpg')
        cat_files = glob(path+'/'+subset+'/cats/*.jpg')
        self.images = dog_files + cat_files
        self.labels = [0]*len(dog_files) + [1]*len(cat_files)
        self.transforms_ = transforms_
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        images = Image.open(self.images[idx]).convert('RGB')
        labels = self.labels[idx]
        if self.transforms_ is not None:
            images = self.transforms_(images)
        return images, labels