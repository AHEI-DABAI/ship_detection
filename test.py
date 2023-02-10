import re
from collections import Counter

label = 'chair 0.44\nperson 0.45\ndining table 0.51\nperson 0.52\nchair 0.54\nperson 0.55\nchair 0.56\nperson 0.58\nchair 0.59\nperson 0.63\nperson 0.64\nchair 0.66\nchair 0.66\nchair 0.72\nperson 0.72\nperson 0.73\nperson 0.74\nchair 0.77\nchair 0.78\nperson 0.78\nperson 0.79\nchair 0.79\nperson 0.82\nperson 0.83\nperson 0.90'
names = [re.sub(r'\d+.\d+', '', content) for content in label.split('\n')]
dic = Counter(names)
content = [key + str(value) for key, value in dic.items()]
content = '\n'.join(content)
print(content)
print(label)

dic = {}

dic.setdefault(re.sub(r'\d+.\d+', '', label), 0) + 1


total_label = ''

total_label += re.sub(r'\d+.\d+', '', label)

