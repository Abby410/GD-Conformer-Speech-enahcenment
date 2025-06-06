import torch.utils.data as data
import torch
import h5py



class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('noisy_signals')
        self.target = hf.get('clean_signals')
        # print(self.data.shape)
        # print(self.target.shape)


    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:]).float(), torch.from_numpy(self.target[index,:,:]).float()


    def __len__(self):
        return self.data.shape[0]
