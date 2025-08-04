import os
import shutil
import json

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)