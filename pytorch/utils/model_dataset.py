import torch
import os
from glob import glob

class AlertDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, num_classes, transform=None):
        '''
        Args:
            root_dir (string): Directory with all of the images
            transform (calllable, optional): Optional transforms that can be 
                applied to the images. 
        '''
        # Initalize the class attributes
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.paths = None
        
        # Get the file names
        self.__get_file_list()
        
        return
    
    def __get_file_list(self):
        '''Get a list of the tensors in the target directory.'''
        
        # Glob all of the tiffs in the root dir
        self.paths = glob(self.root_dir + "//*.pt")
        
        if len(self.paths) == 0:
            raise ValueError("Dataset found no `.pt` files in specified directory.")
        
        return None

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index of tensor to retrieve
        
        Returns:
            torch.tensor: DL Tensor [row, col, band]
        
        '''  
        # if the idx provided is a tensor convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load in the image
        tensor_path = os.path.join(self.root_dir, self.paths[idx])
        
        # Load the tensor
        tensor_2d = torch.load(tensor_path)
            
        # Apply the transform if needed
        if self.transform is not None: 
            tensor_2d = self.transform(tensor_2d)
            
        # create the input tensor and the target labels
        x = tensor_2d[0:-1,:,:]
        y = tensor_2d[-1:,:,:]
        
        return x, y.squeeze(0).long()

    def __len__(self):
        return len(self.paths)
