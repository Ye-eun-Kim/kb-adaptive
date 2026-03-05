"""Default config for KET-QA Retriever (paper Section 5.1, Appendix C)."""

# Paths (override via env or args)
DATA_ROOT = "dataset_ketqa"
TABLE_DIR = None
ENTITY_BASE_DIR = None
QA_DATA_DIR = None

def set_data_paths(root: str):
    global TABLE_DIR, ENTITY_BASE_DIR, QA_DATA_DIR
    TABLE_DIR = f"{root}/tables"
    ENTITY_BASE_DIR = f"{root}/entity_base"
    QA_DATA_DIR = f"{root}/data"


# Common (paper Appendix C.2)
LEARNING_RATE = 1e-5
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
LINEAR_SCHEDULER = True
WARMUP_RATIO = 0.1
DROPOUT_RATE = 0.1

# Bi-encoder (paper: 20 epochs, batch 16, kNS n=25)
BI_ENCODER_EPOCHS = 20
BI_ENCODER_BATCH_SIZE = 16
BI_ENCODER_N_NEGATIVES = 25
BI_ENCODER_MODEL = "bert-base-uncased"

# Cross-encoder (paper: 5 epochs, batch 32, random n=50)
CROSS_ENCODER_EPOCHS = 5
CROSS_ENCODER_BATCH_SIZE = 32
CROSS_ENCODER_N_NEGATIVES = 50
CROSS_ENCODER_MODEL = "roberta-base"

# Inference
N_RETRIEVED_TRIPLES = 200  # paper: N=200

# Special tokens (for serialization)
ADD_SPECIAL_TOKENS = True
