# %%
import os
from os import listdir
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# %%
train_people = ['DennisPNoGlassesGrey', 'JohnGrey', 'SimonBGrey', 'SeanGGrey', 'DanJGrey', 'AdamBGrey', 'JackGrey',
                'RichardHGrey', 'YongminYGrey', 'TomKGrey', 'PaulVGrey', 'DennisPGrey', 'CarlaBGrey', 'JamieSGrey',
                'KateSGrey', 'DerekCGrey', 'KatherineWGrey', 'ColinPGrey', 'SueWGrey', 'GrahamWGrey', 'KrystynaNGrey',
                'SeanGNoGlassesGrey', 'KeithCGrey', 'HeatherLGrey']
test_people = ['RichardBGrey', 'TasosHGrey', 'SarahLGrey', 'AndreeaVGrey', 'YogeshRGrey']

train_phase_train_tuple = {'data': np.ndarray(shape=(len(train_people), 133, 100, 100, 3), dtype=np.uint8),
                           'labels': np.ndarray(shape=(len(train_people), 133), dtype=np.float32)}

test_phase_test_tuple = {'data': np.ndarray(shape=(len(test_people), 133, 100, 100, 3), dtype=np.uint8),
                          'labels': np.ndarray(shape=(len(test_people), 133), dtype=np.float32)}

# %%
root_folder = './data/QMUL/images'

# %%
for i in tqdm(range(len(train_people))):
    one_person_folder = os.path.join(root_folder, train_people[i])
    image_path_list = [f for f in listdir(one_person_folder)]
    for j in range(min(len(image_path_list), 133)):
        image_path = image_path_list[j]
        img_pil = Image.open(os.path.join(one_person_folder, image_path)).convert('RGB')
        img_np = np.asarray(img_pil)
        split_file_name = image_path.split('.')[0].split('_')
        pitch = float(split_file_name[-2])
        angle = float(split_file_name[-1])
        pitch_norm = 2 * ((pitch - 60) / (120 - 60)) - 1
        angle_norm = 2 * ((angle - 0) / (180 - 0)) - 1

        train_phase_train_tuple['data'][i][j] = img_np
        train_phase_train_tuple['labels'][i][j] = pitch_norm

# open a file, where you ant to store the data
pickle_file = open('./data/QMUL/qmul_train.pickle', 'wb')
pickle.dump(train_phase_train_tuple, pickle_file)
pickle_file.close()

# %%
for i in tqdm(range(len(test_people))):
    one_person_folder = os.path.join(root_folder, test_people[i])
    image_path_list = [f for f in listdir(one_person_folder)]
    for j in range(min(len(image_path_list), 133)):
        image_path = image_path_list[j]
        img_pil = Image.open(os.path.join(one_person_folder, image_path)).convert('RGB')
        img_np = np.asarray(img_pil)
        split_file_name = image_path.split('.')[0].split('_')
        pitch = float(split_file_name[-2])
        angle = float(split_file_name[-1])
        pitch_norm = 2 * ((pitch - 60) / (120 - 60)) - 1
        angle_norm = 2 * ((angle - 0) / (180 - 0)) - 1

        test_phase_test_tuple['data'][i][j] = img_np
        test_phase_test_tuple['labels'][i][j] = pitch_norm

# open a file, where you ant to store the data
pickle_file = open('./data/QMUL/qmul_test.pickle', 'wb')
pickle.dump(test_phase_test_tuple, pickle_file)
pickle_file.close()
