import os
from torch.utils.data import Dataset
from PIL import Image


class CustomTensorDataset(Dataset):
    def __init__(self, descriptions, tokenizer, path, transform_images=None):
        self.descriptions = descriptions

        self.links = {}
        for file in os.listdir(path):
            self.links[int(file.split('.')[0])] = path + '/' + file

        self.tokenizer = tokenizer
        self.transform_images = transform_images

    def __getitem__(self, index):

        text = self.descriptions.iloc[index]['description']
        idx = self.descriptions.iloc[index]['id']
        tokenized_text = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,
                                   return_tensors="pt")

        image = Image.open(self.links[idx])
        if self.transform_images:
            image = self.transform_images(image)

        return tokenized_text, image

    def __len__(self):
        return len(self.links)

    