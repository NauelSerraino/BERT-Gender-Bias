# Imports
import numpy as np
import pandas as pd
import os
import re

import sys
sys.path.append('/home/nauel/bert_gender_bias')

from pipelines.utils.paths import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

# Male/Female Dataset

if __name__ == "__main__":
    male_words = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, "male.txt"), header=None)
    male_words.columns = ['word']
    male_words['gender_binary'] = 0

    female_words = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, "female.txt"), header=None)
    female_words.columns = ['word']
    female_words['gender_binary'] = 1

    words = pd.concat([male_words, female_words])
    words.to_csv(os.path.join(INTERIM_DATA_DIR, 'gender_binary_words.csv'), index=False, sep="|")

