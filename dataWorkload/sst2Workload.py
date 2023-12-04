
from os import path, makedirs, listdir, remove
import wget
import shutil
import pandas as pd
from zipfile import ZipFile
from datasets import Dataset

from transformers import AutoTokenizer
import torch
import transformers
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader

class sst2():
    def __init__(self):
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
        test_data = pd.read_csv(self.dataset_path+'test.tsv', sep='\t')

        self.sequence_length = max([len(train_data.iloc[i][0].split(' ')) 
                                    for i in range(len(train_data))])
        
        self.label_list = ['0','1']
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", 
                                                       do_lower_case=False)
        
        
        
        #label_train = train_data[train_data.columns.values[1]].values.tolist() 
        #label2hot = {
        #    0: [1,0],
        #    1: [0,1]
        #    }
        #train_data[train_data.columns.values[1]] = [
        #    label2hot[l] for l in label_train]
        #self.train_dataset = Dataset.from_dict(train_data).with_format("torch")
        
        #self.train_dataset = self.train_dataset.map(self.tokenize_function)
        
        self.train_dataset = self._features_to_dataset(
            self._df_to_features(train_data, "train"))
        self.test_dataset = self._features_to_dataset(
            self._df_to_features(test_data, "test"))
        
        
    def _create_examples(self, df, set_type):
        
        examples = []
        for index, row in df.iterrows():
           
            guid = f"{index}-{set_type}"
            examples.append(
                InputExample(guid=guid, text_a=row['sente'],  label=row['label']))
        return examples
    
    def _df_to_features(self, df, set_type):
       
        examples = self._create_examples(df, set_type)

        #backward compatibility with older transformers versions
        legacy_kwards = {}
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("2.9.0"):
            legacy_kwards = {
                "pad_on_left": False,
                "pad_token": self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                "pad_token_segment_id": 0,
            }

        return glue_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            label_list=self.label_list,
            max_length=self.sequence_length,
            output_mode="classification",
            **legacy_kwards,
        )
    
    def _features_to_dataset(self, features):
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
        return dataset
    
    def get_dataloader (self, batch_size, set_type = 'train'):
        if set_type == 'train':
            return DataLoader(self.train_dataset, batch_size=batch_size)
        elif set_type == 'test':
            return DataLoader(self.test_dataset, batch_size=batch_size)
    
    #def tokenize_function(self, examples):
    #    return self.tokenizer(examples["sentence"], truncation=True, 
    #                     padding="max_length", max_length=self.sequence_length)
    
   
    