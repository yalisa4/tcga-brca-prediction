# --- Load data ---
import pandas as pd

expre = pd.read_csv('data/tcga_brca_expression_raw.tsv', sep='\t')
pheno = pd.read_csv('data/tcga_brca_phenotypes.tsv', sep='\t')

# --- Clean & filter ---
# --- Replace missing receptor status values with secondary source ---
pheno["breast_carcinoma_estrogen_receptor_status"] = pheno["breast_carcinoma_estrogen_receptor_status"].fillna(
    pheno["ER_Status_nature2012"]
)
pheno["breast_carcinoma_progesterone_receptor_status"] = pheno["breast_carcinoma_progesterone_receptor_status"].fillna(
    pheno["PR_Status_nature2012"]
)

# --- Remove rows where both receptor statuses are missing ---
mask_both_nan = (
    pheno["breast_carcinoma_estrogen_receptor_status"].isna() &
    pheno["breast_carcinoma_progesterone_receptor_status"].isna()
)
pheno = pheno[~mask_both_nan]

# --- Filter for valid sample types ---
pheno = pheno[pheno["sample_type"].isin(["Primary Tumor", "Solid Tissue Normal"])]

# --- Remove ambiguous or missing values ---
pheno = pheno[
    (pheno["breast_carcinoma_estrogen_receptor_status"] != "Indeterminate") &
    (pheno["breast_carcinoma_progesterone_receptor_status"] != "Indeterminate") &
    (~pheno["gender"].isna()) &
    (pheno["HER2_Final_Status_nature2012"] != "Equivocal")
]

# --- 3. Preprocessing / scaling ---

# --- 4. Split into train/test ---

# --- 5. Train model ---

# --- 6. Evaluate ---