import re
import json
from sklearn.model_selection import train_test_split

""" Build a TextDataset using Instructions in recipes.json """
with open("recipes.json") as file:
    data = json.load(file)


def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w')
    dataset = ''
    for texts in data_json:
        summary = str(texts['Instructions']).strip()
        summary = re.sub(r"\s", " ", summary)
        dataset += summary + " "
    f.write(dataset)


train, test = train_test_split(data, test_size=0.15)

build_text_files(train, 'train_dataset.txt')
build_text_files(test, 'test_dataset.txt')

print("Train dataset length: ", str(len(train)))
print("Test dataset length: ", str(len(test)))
