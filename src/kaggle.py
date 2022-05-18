import os
os.system('rocminfo')
os.environ['KAGGLE_USERNAME'] = "kirillsr" # username from the json file
os.environ['KAGGLE_KEY'] = "4476f1ad99a65fc5718a0781e5f4c5aa" # Provide your key from the json file
# os.system('kaggle competitions download -c dogs-vs-cats')

from zipfile import ZipFile

file_name = "./train.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')

# !kaggle competitions download -c footballteams # api copied from kaggle