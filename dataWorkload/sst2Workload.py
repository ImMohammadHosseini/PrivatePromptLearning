
from os import path, makedirs, listdir, remove
import wget
import shutil
import pandas as pd
from zipfile import ZipFile


class sst2():
    def __init__(self):
        super().__init__()
        dataset_path = 'dataWorkload/datasets/SST-2/'
        if not path.exists(dataset_path):
            #makedirs(dataset_path)
            print('Downloading sst2 Dataset')
            
            url = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'
            filename = wget.download(url)
            
            zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
            remove(filename)        
        self.dataset_path = dataset_path+'SST-2/'
        train_data = pd.read_csv(self.dataset_path+'train.tsv', sep='\t')
        self.sequence_length = max([len(train_data.iloc[i][0].split(' ')) for i in range(len(train_data))])
        
    def read_csv (self):
        pass
        