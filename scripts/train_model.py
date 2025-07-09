# --- Load data ---
import pandas as pd

expre = pd.read_csv('data/tcga_brca_expression_raw.tsv', sep='\t')
pheno = pd.read_csv('data/tcga_brca_phenotypes.tsv', sep='\t')

# --- Clean & filter ---
pheno = pheno[pheno["sample_type"].isin(["Primary Tumor", "Solid Tissue Normal"])]

pheno = pheno.dropna(subset=[
    "Gender_nature2012",
    "Age_at_Initial_Pathologic_Diagnosis_nature2012",
    "ER_Status_nature2012",
    "PR_Status_nature2012",
    "HER2_Final_Status_nature2012"
])

pheno["label"] = pheno["sample_type"].map({
    "Primary Tumor": 1,
    "Solid Tissue Normal": 0
})

selected_cols = [
    "Gender_nature2012",
    "Age_at_Initial_Pathologic_Diagnosis_nature2012",
    "ER_Status_nature2012",
    "PR_Status_nature2012",
    "HER2_Final_Status_nature2012",
    "label"
]

pheno = pheno[selected_cols].copy()

expre_T = expre.T

common_samples = expre_T.index.intersection(pheno.index)
expre_T = expre_T.loc[common_samples]
pheno = pheno.loc[common_samples]

# --- 3. Preprocessing / scaling ---

# --- 4. Split into train/test ---

# --- 5. Train model ---

# --- 6. Evaluate ---