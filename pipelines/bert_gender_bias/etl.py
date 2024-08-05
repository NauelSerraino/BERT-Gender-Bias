import pandas as pd
import os

from pipelines.utils.paths import EXTERNAL_DATA_DIR

gendered_words = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'gendered_words.txt'), header=None, sep='\t')
dictionary = dict(zip(gendered_words[0], gendered_words[1]))

import pandas as pd

class PreprocessWinoBias:
    def __init__(self, dictionary, keys_to_swap):
        self.dictionary = dictionary
        self.keys_to_swap = keys_to_swap

    def clean(self):
        swapped_dict = self._swap_keys_values(self.dictionary, self.keys_to_swap)
        unique_dict = self._remove_duplicates(swapped_dict)
        self.df = self._dict_to_dataframe(unique_dict)
        return self.df

    def _normalize_pair(self, pair):
        return tuple(sorted(pair))

    def _remove_duplicates(self, d):
        normalized = set()
        unique_dict = {}

        for k, v in d.items():
            norm_kv = self._normalize_pair((k.strip(), v.strip()))
            if norm_kv not in normalized:
                normalized.add(norm_kv)
                unique_dict[k.strip()] = v.strip()

        return unique_dict

    def _swap_keys_values(self, d, keys_to_swap):
        result = dict(d)
        swap_pairs = {}

        for key in keys_to_swap:
            if key in d:
                value = d[key].strip()
                if value not in swap_pairs:
                    swap_pairs[value] = key.strip()
        
        for k, v in swap_pairs.items():
            result[v] = k
        
        for key in keys_to_swap:
            if key in result:
                del result[key]
        
        return result

    def _dict_to_dataframe(self, d):
        males = list(d.keys())
        females = list(d.values())

        df = pd.DataFrame({
            'male': males,
            'female': females
        })

        return df


keys_to_swap = [
    'aunt', 
    'aunts', 
    'bride', 
    'brides', 
    'chairwomen', 
    'chick', 
    'chicks', 
    'daughter', 
    'daughters', 
    'female', 
    'females', 
    'gal',
    'gals',
    'granddaughter',
    'granddaughters',
    'herself',
    'lady',
    "ma'am",
    'miss',
    'ms.'
    ]

preprocess = PreprocessWinoBias(dictionary, keys_to_swap)
df = preprocess.clean()

