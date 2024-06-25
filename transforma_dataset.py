'''
Script q deve ser rodado apenas uma vez para mudar o dataset
do formato do Roboflow com 16 tipos de formigas para classificação
binária (tem ou não formiga).

Dataset: roboflow djay-de-gier-fopbf ant-object-detection
'''


import os

pastas = ["train", "valid", "test"]
dataset_folder = 'Ant_object_detection'

for pasta in pastas:
    arquivos = os.listdir(f"datasets/{dataset_folder}/{pasta}/labels")
    arquivos = [arquivo for arquivo in arquivos if arquivo.endswith('.txt')]
    for arquivo in arquivos:
        with open(f'datasets/{dataset_folder}/{pasta}/labels/{arquivo}', 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                if line[1] != ' ':  # caso classe tenha dois dígitos (10, 11,...)
                    line = '0' + line[2:]
                else:
                    line = '0' + line[1:]
                file.write(line)
            file.truncate()