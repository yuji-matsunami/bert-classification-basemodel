from transformers import BertTokenizer
import pandas as pd
import os
import json

class Dataloader:
    def __init__(self, dir_name) -> None:
        self.files_list = self.get_jsonfiles(dir_name)
        self.df_jsons = self.make_dataflame(dir_name,self.files_list)
    
    def get_jsonfiles(self, dir_name):
        files_list = []
        for file_name in os.listdir(dir_name):
            if ".json" in file_name:
                files_list.append(file_name)
        return files_list

    def make_dataflame(self, dir_name, files_list):
        with open(dir_name+'/'+files_list[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_jsons = pd.DataFrame(data["sentences"])
        del files_list[0]
        for file_name in files_list:
            file_path = dir_name+'/'+file_name
            with open(file_path,'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data["sentences"])
            pd.concat([df_jsons, df])
        return df_jsons

    
    def test(self):
        # print(self.df_csv)
        print(self.files_list)
        print(self.df_jsons)

def main():
    dataset = Dataloader(dir_name="../../chABSA-dataset")
    dataset.test()


if __name__ == '__main__':
    main()