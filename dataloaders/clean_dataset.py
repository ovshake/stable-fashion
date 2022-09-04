import glob
import os
import json

def write_txt(list_of_ids, filename):
    assert os.splitext(filename)[1] == 'txt'
    with open(filename, 'w') as f:
        for ids in list_of_ids:
            f.write(f"{ids}\n")

    print('writing done!')




def fetch_all_shop_images(annotation_folders):
