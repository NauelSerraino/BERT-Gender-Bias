import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, '0_external')
RAW_DATA_DIR = os.path.join(DATA_DIR, '0_raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, '1_interim')
FINAL_DATA_DIR = os.path.join(DATA_DIR, '2_final')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
FEATURES_MODEL_DIR = os.path.join(OUTPUT_DIR, 'features')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'reports', 'figures')

