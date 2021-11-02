from pandas.core.frame import DataFrame
from transformers import BertTokenizer
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

class Dataloader:
    def __init__(self, dir_name) -> None:
        self.files_list = self.get_jsonfiles(dir_name)
        self.df_jsons = self.make_dataflame(dir_name,self.files_list)
        self.binary_classes_df = self.ngposi_dataflame(self.df_jsons)
    
    def get_jsonfiles(self, dir_name):
        """[summary]
        ディレクトリにあるjsonファイル名を取得して
        リストにする
        Args:
            dir_name ([type] string): [description] ディレクトリ名

        Returns:
            [type []string]: [description] jsonファイル名のリスト
        """
        files_list = []
        for file_name in os.listdir(dir_name):
            if ".json" in file_name:
                files_list.append(file_name)
        return files_list

    def ngposi_dataflame(self, dataflame):
        """[summary]
        Create a data frame for binary classification

        Args:
            dataflame ([type]pands.DataFrame): [description] Dataframes for all chABSA data sets
        
        Returns:
            [type pd.DataFrame]: [defcription] textとlabel(positive:0, negative:1)が入ったデータフレーム
        """
        texts = []
        labels = []

        for sentence, opinions in zip(dataflame['sentence'], dataflame['opinions']):
            polarity = 0
            for opinion in opinions:
                if opinion["polarity"] == "positive":
                    polarity += 1
                elif opinion["polarity"] == "negative":
                    polarity -= 1

            if polarity == 0: continue

            if polarity > 0:
                label = 0
            elif polarity < 0:
                label = 1
            # print(label)
            texts.append(sentence)
            labels.append(label)
        two_classes_df = pd.DataFrame({'texts': texts, 'labels':labels})
        # print("DataFrame Shape:", two_classes_df.shape)
        return two_classes_df
        

    def make_dataflame(self, dir_name, files_list):
        """[summary]
        学習用dataframe作成
        Args:
            dir_name ([type]): [description]
            files_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        with open(dir_name+'/'+files_list[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_jsons = pd.DataFrame(data["sentences"])
        del files_list[0]
        print("file num:", len(files_list))
        for file_name in files_list:
            file_path = dir_name+'/'+file_name
            with open(file_path,'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data["sentences"])
            df_jsons = pd.concat([df_jsons, df])
        print("DataFrame Shape", df_jsons.shape)
        # self.ngposi_dataflame(df_jsons)
        return df_jsons

    
    def test(self):
        """[summary]
        test用
        """
        # print(self.df_csv)
        # print(self.files_list)
        # print(self.df_jsons)

    def binary_classfication_dataset(self, train_size=0.8):
        """[summary]
        - obtain a data frame for binary classification
        - split data return for trianing and testing
        Args:
            train_size ([type]float): [description] should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split
        Returns:
            [type]pandas.DataFrame: [description] dataframe texts:, label:(posi:0,neg:1)
        """
        train_df, test_df = train_test_split(self.binary_classes_df, train_size=train_size, random_state=42)

        return train_df, test_df

def test_binary_dataset(dataframe:DataFrame):
    cnt = 0
    for row in dataframe.itertuples():
        print(row)
        cnt += 1
        if cnt == 10:
            break
    print("------")
    print(dataframe.size)


def main():
    dataset = Dataloader(dir_name="../../chABSA-dataset")
    train_df, test_df = dataset.binary_classfication_dataset()
    test_binary_dataset(train_df)
    test_binary_dataset(test_df)
    # dataset.test()


if __name__ == '__main__':
    main()