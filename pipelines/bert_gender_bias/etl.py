
#%% Imports
import numpy as np
import pandas as pd
import os
import re

import sys
sys.path.append('/home/nauel/bert_gender_bias')

from pipelines.utils.paths import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

#%% Preprocess WinoBias
class PreprocessWinoBias:
    def __init__(self, dictionary, keys_to_swap):
        self.dictionary = dictionary
        self.keys_to_swap = keys_to_swap

    def clean(self):
        unique_dict = self._remove_duplicates(self.dictionary)
        swapped_dict = self._swap_keys_values(unique_dict, self.keys_to_swap)
        self.df = self._dict_to_dataframe(swapped_dict)
        self.df = self._melt_and_more()
        return self.df
    
    def _normalize_pair(self, pair):
        """Return a canonical form of the pair (sorted tuple)."""
        return tuple(sorted(pair))

    def _remove_duplicates(self, d):
        """Remove duplicates from the dictionary considering swapped pairs."""
        normalized = set()
        unique_dict = {}

        for k, v in d.items():
            norm_kv = self._normalize_pair((k.strip(), v.strip()))
            if norm_kv not in normalized:
                normalized.add(norm_kv)
                unique_dict[k.strip()] = v.strip()

        return unique_dict

    def _swap_keys_values(self, d, keys_to_swap):
        """Swap key-value pairs in the dictionary for specific keys."""
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

    def _melt_and_more(self):
        df = self.df.melt()
        df['gender_binary'] = np.where(df.variable == 'female', 1, 0)
        df.columns = ['gender', 'word', 'gender_binary']
        df = df.sort_values(by='gender_binary', ascending=True).reset_index(drop=True)
        return df


class PreprocessJobs():
    def __init__(self, df):
        self.df = df
    
    def clean(self):
        self.df['job_title_clean'] = self.df['job_title'].apply(lambda x: re.sub(r'\b\d+\w*\b', '', x))
        job_df = pd.DataFrame()
        job_df['job_title_clean'] = self.df['job_title_clean'].str.split('/').explode().reset_index(drop=True)

        job_df['n_tokens'] = job_df.job_title_clean.apply(lambda x: len(x.split()))
        job_df = job_df[job_df.n_tokens==1].reset_index(drop=True)
        job_df.drop_duplicates(subset="job_title_clean", inplace=True)
        
        job_df['job_title_clean'].replace(' ', '', inplace=True)
        job_df = job_df[job_df['job_title_clean'].str.len() >= 4].reset_index(drop=True)
        job_df.reset_index(drop=True, inplace=True)
        
        return job_df
        
    
#%% Gendered Words Dataset
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
    'ms.',
    
    ]

gendered_words = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'gendered_words.txt'), header=None, sep='\t')
dictionary = dict(zip(gendered_words[0], gendered_words[1]))

preprocess = PreprocessWinoBias(dictionary, keys_to_swap)
df_gendered = preprocess.clean()
print(df_gendered)

# %% Extra Gendered Words Dataset
keys_to_swap = ['camerawomen']

gendered_words = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'extra_gendered_words.txt'), header=None, sep='\t')
dictionary = dict(zip(gendered_words[0], gendered_words[1]))

preprocess = PreprocessWinoBias(dictionary, keys_to_swap)
df_extra_gendered = preprocess.clean()
print(df_extra_gendered.head(60))
    

# %% Save to CSV
gendered_words_df = pd.concat([df_gendered, df_extra_gendered])
gendered_words_df = gendered_words_df.sort_values(by='gender_binary', ascending=True).reset_index(drop=True)
print(gendered_words_df)

gendered_words_df.to_csv(os.path.join(INTERIM_DATA_DIR, 'gendered_words.csv'), index=False, sep="|")


# %% Jobs Dataset
jobs = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'job_titles.txt'), header=None, sep='\t', names=['job_title'])
preprocess = PreprocessJobs(jobs)

df_jobs = preprocess.clean()

df_jobs.to_csv(os.path.join(INTERIM_DATA_DIR, 'jobs.csv'), index=False, sep="|")


# %%
df_jobs[df_jobs['job_title_clean'].str.len() < 5]

# %%
