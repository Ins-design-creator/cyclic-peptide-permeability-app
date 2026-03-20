
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

st.set_page_config(page_title="Cyclic Peptide Permeability Predictor", page_icon="🧪", layout="wide")

DEFAULT_DATA = "cyclic_peptide_final_outputs_plus_validated100.csv"

STAGE1_FEATURES = [
    "mw_rdkit",
    "tpsa_rdkit",
    "hba_rdkit",
    "hbd_rdkit",
    "rotatable_bonds_rdkit",
    "clogp_rdkit",
    "heavy_atom_count_rdkit",
    "fraction_csp3_rdkit",
    "ring_count_rdkit",
    "formal_charge_rdkit",
    "nhoh_count_rdkit",
    "no_count_rdkit",
]

STAGE2_FEATURES = ["pred_proxy", "tpsa", "rotatable_bonds"]


def calc_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES.")
    return {
        "mw_rdkit": Descriptors.MolWt(mol),
        "tpsa_rdkit": rdMolDescriptors.CalcTPSA(mol),
        "hba_rdkit": rdMolDescriptors.CalcNumHBA(mol),
        "hbd_rdkit": rdMolDescriptors.CalcNumHBD(mol),
        "rotatable_bonds_rdkit": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "clogp_rdkit": Crippen.MolLogP(mol),
        "heavy_atom_count_rdkit": mol.GetNumHeavyAtoms(),
        "fraction_csp3_rdkit": rdMolDescriptors.CalcFractionCSP3(mol),
        "ring_count_rdkit": rdMolDescriptors.CalcNumRings(mol),
        "formal_charge_rdkit": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "nhoh_count_rdkit": Lipinski.NHOHCount(mol),
        "no_count_rdkit": Lipinski.NOCount(mol),
        # model-2 passthrough features
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


@st.cache_data
def load_training_data(uploaded_file_bytes=None, uploaded_name=None):
    if uploaded_file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(uploaded_file_bytes))
        source = uploaded_name or "uploaded dataset"
    else:
        path = Path(DEFAULT_DATA)
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {DEFAULT_DATA}. Put the corrected CSV next to this app or upload it in the sidebar."
            )
        df = pd.read_csv(path)
        source = str(path)
    return df, source


@st.cache_resource
def train_models(df_hash_key, _df):
    df = _df.copy()
    df["PAMPA_numeric"] = pd.to_numeric(df["PAMPA_numeric"], errors="coerce")

    desc_rows = []
    for s in df["smiles"]:
        desc_rows.append(calc_descriptors(s))
    desc_df = pd.DataFrame(desc_rows)

    model_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)

    use_cols = STAGE1_FEATURES + ["tpsa", "rotatable_bonds", "PAMPA_numeric", "stage4_exposed_polarity_proxy"]
    model_df = model_df[use_cols].dropna().copy()

    train_idx, test_idx = train_test_split(model_df.index, test_size=0.2, random_state=42)
    train = model_df.loc[train_idx].copy()
    test = model_df.loc[test_idx].copy()

    stage1 = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2)
    stage1.fit(train[STAGE1_FEATURES], train["stage4_exposed_polarity_proxy"])
    train["pred_proxy"] = stage1.predict(train[STAGE1_FEATURES])
    test["pred_proxy"] = stage1.predict(test[STAGE1_FEATURES])

    stage2 = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=3)
    stage2.fit(train[STAGE2_FEATURES], train["PAMPA_numeric"])
    test_pred = stage2.predict(test[STAGE2_FEATURES])

    train["high_perm"] = train["PAMPA_numeric"] > -6
    test["high_perm"] = test["PAMPA_numeric"] > -6

    clf = RandomForestClassifier(n_estimators=400, random_state=42, min_samples_leaf=3)
    clf.fit(train[STAGE2_FEATURES], train["high_perm"])
    test_prob = clf.predict_proba(test[STAGE2_FEATURES])[:, 1]

    metrics = {
        "stage1_r2": r2_score(test["stage4_exposed_polarity_proxy"], test["pred_proxy"]),
        "stage1_mae": mean_absolute_error(test["stage4_exposed_polarity_proxy"], test["pred_proxy"]),
        "stage2_r2": r2_score(test["PAMPA_numeric"], test_pred),
        "stage2_mae": mean_absolute_error(test["PAMPA_numeric"], test_pred),
        "stage2_auc": roc_auc_score(test["high_perm"], test_prob),
    }

    feature_importance = pd.Series(stage2.feature_importances_, index=STAGE2_FEATURES).sort_values(ascending=False)
    return stage1, stage2, clf, metrics, feature_importance


def predict_smiles(smiles, stage1, stage2, clf):
    desc = calc_descriptors(smiles)
    desc_df = pd.DataFrame([desc])

    pred_proxy = float(stage1.predict(desc_df[STAGE1_FEATURES])[0])
    stage2_input = pd.DataFrame([{
        "pred_proxy": pred_proxy,
        "tpsa": desc["tpsa"],
        "rotatable_bonds": desc["rotatable_bonds"],
    }])

    pred_logpapp = float(stage2.predict(stage2_input)[0])
    high_perm_prob = float(clf.predict_proba(stage2_input)[0, 1])
    pred_papp = 10 ** pred_logpapp

    return {
        "pred_proxy": pred_proxy,
        "pred_logpapp": pred_logpapp,
        "pred_papp_cm_s": pred_papp,
        "high_perm_prob": high_perm_prob,
        "descriptors": desc,
    }


st.title("🧪 Cyclic Peptide Permeability Predictor")
st.write(
    "Two-stage model trained from your corrected dataset. "
    "Stage 1 predicts the exposed-polarity proxy from SMILES-derived descriptors. "
    "Stage 2 predicts log10(Papp, cm/s) from predicted proxy, TPSA, and rotatable bonds."
)

with st.sidebar:
    st.header("Training data")
    uploaded = st.file_uploader("Optional: upload corrected training CSV", type=["csv"])
    st.caption("If you do not upload a file, the app looks for the corrected CSV next to the app.")

    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    uploaded_name = uploaded.name if uploaded is not None else None

df, source = load_training_data(uploaded_bytes, uploaded_name)
df_hash_key = f"{len(df)}_{hash(tuple(df.columns))}_{source}"
stage1, stage2, clf, metrics, feature_importance = train_models(df_hash_key, df)

col1, col2, col3 = st.columns(3)
col1.metric("Stage 1 R²", f"{metrics['stage1_r2']:.3f}")
col2.metric("Stage 2 R²", f"{metrics['stage2_r2']:.3f}")
col3.metric("High-perm ROC AUC", f"{metrics['stage2_auc']:.3f}")

with st.expander("Model details", expanded=False):
    st.write(f"Training source: `{source}`")
    st.write("Stage 2 feature importance:")
    st.dataframe(
        pd.DataFrame(
            {"feature": feature_importance.index, "importance": feature_importance.values}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Note: predicted proxy and TPSA are correlated, so feature importance should be interpreted cautiously."
    )

st.subheader("Predict from SMILES")
default_smiles = df["smiles"].dropna().iloc[0]
smiles = st.text_area("Paste cyclic peptide SMILES", value=default_smiles, height=140)

if st.button("Predict permeability", type="primary"):
    try:
        result = predict_smiles(smiles, stage1, stage2, clf)

        a, b, c = st.columns(3)
        a.metric("Predicted log10(Papp, cm/s)", f"{result['pred_logpapp']:.3f}")
        b.metric("Predicted Papp (cm/s)", f"{result['pred_papp_cm_s']:.2e}")
        c.metric("Prob. high permeability (> -6)", f"{100*result['high_perm_prob']:.1f}%")

        st.write("### Predicted internal proxy")
        st.write(f"Exposed polarity proxy: **{result['pred_proxy']:.3f}**")

        desc_table = pd.DataFrame(
            {"descriptor": list(result["descriptors"].keys()), "value": list(result["descriptors"].values())}
        )
        st.write("### SMILES-derived descriptors used")
        st.dataframe(desc_table, hide_index=True, use_container_width=True)

        if result["pred_logpapp"] > -6:
            st.success("This peptide lands in the higher-permeability regime by the current classifier.")
        else:
            st.warning("This peptide lands in the lower-permeability regime by the current classifier.")

    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption(
    "Use as a prioritization tool, not a definitive assay replacement. "
    "Performance depends on how similar new chemistry is to the training set."
)
