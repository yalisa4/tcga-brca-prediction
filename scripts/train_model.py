# --- Load data ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# --- Preprocessing / scaling ---
bins = [0, 40, 50, 60, 70, 100]
labels = ['<40', '40-49', '50-59', '60-69', '70+']

pheno['age_group'] = pd.cut(
    pheno['Age_at_Initial_Pathologic_Diagnosis_nature2012'],
    bins=bins,
    labels=labels,
    right=False 
)
pheno['age_group'] = pheno['age_group'].cat.add_categories('Unknown')
pheno['age_group'] = pheno['age_group'].fillna('Unknown')

pheno['HER2_Final_Status_nature2012'] = pheno['HER2_Final_Status_nature2012'].astype('category')
pheno['HER2_Final_Status_nature2012'] = pheno['HER2_Final_Status_nature2012'].cat.add_categories('Unknown')
pheno['HER2_Final_Status_nature2012'] = pheno['HER2_Final_Status_nature2012'].fillna('Unknown') 

expre_T = expre.set_index("sample").T
pheno = pheno.set_index("sampleID")

common_samples = expre_T.index.intersection(pheno.index)
expre_T = expre_T.loc[common_samples]
pheno = pheno.loc[common_samples]

pheno = pheno[[
    "sample_type",
    "gender",
    "age_group",
    "breast_carcinoma_estrogen_receptor_status",
    "breast_carcinoma_progesterone_receptor_status",
    "HER2_Final_Status_nature2012"
]]

# --- Normalisation ---
scaler = StandardScaler()
expre_scaled = scaler.fit_transform(expre_T)
expre_scaled = pd.DataFrame(expre_scaled, index=expre_T.index, columns=expre_T.columns)

# --- 4. Split into train/test ---

# --- 5. Train model ---

# --- 6. Evaluate ---