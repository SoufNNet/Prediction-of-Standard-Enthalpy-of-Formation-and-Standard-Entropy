import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import re
import io
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

try:
 from rdkit import Chem
 from rdkit.Chem import Descriptors, Draw
 from rdkit.ML.Descriptors import MoleculeDescriptors
 RDKIT_OK = True
except ImportError:
 RDKIT_OK = False

R_GAS_CONSTANT = 8.314
TEMPERATURE = 298.15
ADIM_TO_KJMOL = R_GAS_CONSTANT * TEMPERATURE / 1000.0

def init_params(m):
 for _, module in m.named_modules():
  for param_name, param in module.named_parameters():
   if 'weight' in param_name:
    if any(k in param_name for k in ('lin', 'ih')):
     nn.init.xavier_uniform_(param)
    elif 'hh' in param_name:
     nn.init.orthogonal_(param)
   elif param_name == 'bias':
    nn.init.constant_(param, 0.0)

class Vocab:
 def __init__(self, tokens):
  self.itos = tokens
  self.stoi = {t: i for i, t in enumerate(tokens)}
 def __len__(self):
  return len(self.itos)

class Encoder(nn.Module):
 def __init__(self, input_size=48, hidden_size=512, n_layers=2,
     bidirectional=True, latent_size=56):
  super().__init__()
  self.hidden_factor = (2 if bidirectional else 1) * n_layers
  self.rnn = nn.GRU(input_size, hidden_size, n_layers,
       bidirectional=bidirectional, batch_first=True)
  self.mean_lin = nn.Linear(hidden_size * self.hidden_factor, latent_size)
  self.logvar_lin = nn.Linear(hidden_size * self.hidden_factor, latent_size)
  init_params(self)

 def forward(self, x):
  _, h = self.rnn(x)
  h = h.permute(1, 0, 2).contiguous().view(h.size(1), -1)
  return self.mean_lin(h), -torch.abs(self.logvar_lin(h))

class Decoder(nn.Module):
 def __init__(self, input_size=48, hidden_size=512, n_layers=4,
     dropout=0.5, latent_size=56, vocab_size=64):
  super().__init__()
  self.hidden_size = hidden_size
  self.hidden_factor = n_layers
  self.embedding_dropout = nn.Dropout(dropout)
  self.rnn    = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
  self.latent2hidden  = nn.Linear(latent_size, hidden_size * n_layers)
  self.outputs2vocab  = nn.Linear(hidden_size, vocab_size)
  self.outputs_dropout = nn.Dropout(dropout)
  init_params(self)

 def forward(self, emb, z):
  h = self.latent2hidden(z)
  h = torch.tanh(h.view(-1, self.hidden_factor,
        self.hidden_size).permute(1, 0, 2).contiguous())
  emb = self.embedding_dropout(emb)
  out, _ = self.rnn(emb, h)
  b, s, hs = out.size()
  return self.outputs2vocab(self.outputs_dropout(out.view(-1, hs))).view(b, s, -1)

class Vae(nn.Module):
 def __init__(self, vocab_size, embedding_size, dropout, n_layers, hidden_size,
     bidirectional=True, latent_size=56):
  super().__init__()
  self.embedding = nn.Embedding(vocab_size, embedding_size)
  self.encoder = Encoder(embedding_size, hidden_size, n_layers,
         bidirectional, latent_size)
  dec_layers  = (2 if bidirectional else 1) * n_layers
  self.decoder = Decoder(embedding_size, hidden_size, dec_layers,
         dropout, latent_size, vocab_size)

 def encode_to_mean(self, x):
  x = x.cuda() if next(self.parameters()).is_cuda else x
  emb = self.embedding(x)
  mean, _ = self.encoder(emb)
  return mean

class ANNRegressor(nn.Module):
 def __init__(self, input_size, hidden_layers, dropout_rate=0.3):
  super().__init__()
  layers, prev = [], input_size
  for hs in hidden_layers:
   layers += [nn.Linear(prev, hs), nn.ReLU(),
      nn.BatchNorm1d(hs), nn.Dropout(dropout_rate)]
   prev = hs
  layers.append(nn.Linear(prev, 1))
  self.network = nn.Sequential(*layers)

 def forward(self, x):
  return self.network(x).squeeze()

class DescriptorProcessor:
 def __init__(self):
  self.columns_after_dropna = None
  self.variance_selector  = None
  self.final_descriptor_names = None
  self.scaler_desc   = None

 def transform(self, desc_arr, desc_names):
  df  = pd.DataFrame(desc_arr, columns=desc_names)
  df  = df[self.columns_after_dropna]
  df_var = pd.DataFrame(
   self.variance_selector.transform(df),
   columns=df.columns[self.variance_selector.get_support()])
  return self.scaler_desc.transform(df_var[self.final_descriptor_names])

def tokenize_smiles(smiles):
 pattern = r'\[[^\]]+\]|%\d{2}|Br|Cl|se|as|@@|[BCNOPSFIbcnops]|[=#\-+:\/\\().\[\]@]|\d'
 return re.findall(pattern, str(smiles))

def smiles_to_tensor(smiles_list, vocab, max_len=75):
 data = []
 for smi in smiles_list:
  toks = tokenize_smiles(smi)
  idx = ([vocab.stoi['<sos>']] +
    [vocab.stoi.get(t, vocab.stoi['<unk>']) for t in toks])[:max_len]
  idx += [vocab.stoi[' ']] * (max_len - len(idx))
  data.append(torch.LongTensor(idx))
 return torch.stack(data)

def extract_latent(vae, smiles_list, vocab, device, batch=128):
 vae.eval()
 vecs = []
 with torch.no_grad():
  for i in range(0, len(smiles_list), batch):
   x = smiles_to_tensor(smiles_list[i:i+batch], vocab).to(device)
   vecs.append(vae.encode_to_mean(x).cpu().numpy())
 return np.vstack(vecs)

def canonicalize(smi):
 if not RDKIT_OK:
  return smi
 mol = Chem.MolFromSmiles(str(smi))
 return Chem.MolToSmiles(mol, canonical=True) if mol else None

def calc_descriptors(smiles_list, desc_names):
 if not RDKIT_OK:
  return [], [], []
 calc  = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
 rows, vi, errs = [], [], []
 for i, smi in enumerate(smiles_list):
  try:
   mol = Chem.MolFromSmiles(smi)
   if mol:
    d = list(calc.CalcDescriptors(mol))
    if all(np.isfinite(v) for v in d):
     rows.append(d); vi.append(i)
    else:
     errs.append(smi)
   else:
    errs.append(smi)
  except Exception:
   errs.append(smi)
 return rows, vi, errs

def mol_to_image(smi):
 if not RDKIT_OK:
  return None
 try:
  mol = Chem.MolFromSmiles(smi)
  if mol:
   return Draw.MolToImage(mol, size=(220, 160))
 except Exception:
  pass
 return None

@st.cache_resource
def load_pipeline(sklearn_path, ann_path, vae_cfg_path, vae_weights_path):
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 with open(sklearn_path, "rb") as f:
  bundle = pickle.load(f)
 scaler  = bundle["scaler"]
 dp   = bundle["descriptor_processor"]
 desc_names = bundle["safe_descriptor_names"]
 hidden  = bundle["hidden_layers"]

 with open(vae_cfg_path, "rb") as f:
  vb = pickle.load(f)
 vocab = vb["vocab"]
 vcfg = vb["vae_config"]

 ann_ckpt = torch.load(ann_path, map_location=device, weights_only=False)
 input_size = scaler.mean_.shape[0]
 ann  = ANNRegressor(input_size, hidden).to(device)
 ann.load_state_dict({k: v for k, v in ann_ckpt["model_state_dict"].items()
       if not k.endswith((".x", ".y"))}, strict=True)
 ann.eval()

 vae = Vae(
  vocab_size=vcfg["vocab_size"],
  embedding_size=vcfg["embedding_size"], dropout=vcfg["dropout"],
  n_layers=vcfg["n_layers"], hidden_size=vcfg["hidden_size"],
  bidirectional=True, latent_size=vcfg["latent_size"],
 )
 vae_ckpt = torch.load(vae_weights_path, map_location=device, weights_only=False)
 vae.load_state_dict(vae_ckpt["model_state_dict"])
 vae = vae.to(device)
 vae.eval()

 return ann, scaler, dp, vae, vocab, desc_names, device

def predict_smiles(smiles_list, ann, scaler, dp, vae, vocab, desc_names, device, conv):
 canonical = []
 for smi in smiles_list:
  c = canonicalize(smi.strip())
  canonical.append(c)

 raw, vi, errs = calc_descriptors(
  [c for c in canonical if c], desc_names)

 if not vi:
  return pd.DataFrame(), errs

 valid_smi = [c for c in canonical if c]
 valid_smi_f = [valid_smi[i] for i in vi]
 desc_arr = np.array(raw)
 desc_norm = dp.transform(desc_arr, desc_names)
 lat   = extract_latent(vae, valid_smi_f, vocab, device)
 X   = scaler.transform(np.concatenate([lat, desc_norm], axis=1))

 ann.eval()
 with torch.no_grad():
  y = ann(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy() * conv

 df = pd.DataFrame({
  "SMILES (entré)" : [smiles_list[i] for i in vi],
  "SMILES canonique" : valid_smi_f,
  "Valeur prédite" : np.round(y, 2),
 })
 return df, errs

try:
 from streamlit_ketcher import st_ketcher
 KETCHER_OK = True
except ImportError:
 KETCHER_OK = False

st.set_page_config(
 page_title="ΔfH° / S° Prediction",
 page_icon=None,
 layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.page-sub {
 font-size: 13px;
 color:
 margin-bottom: 0;
}
.result-card-h {
 background: #fde8e8;
 border-radius: 12px;
 padding: 22px 28px;
 margin: 8px 0;
}
.result-card-s {
 background: #e6f4ea;
 border-radius: 12px;
 padding: 22px 28px;
 margin: 8px 0;
}
.result-card-h .label { font-size: 13px; color: #a33; margin: 0 0 4px 0; }
.result-card-h .value { font-size: 28px; font-weight: 700; color: #7a0000; margin: 0; }
.result-card-h .unit  { font-size: 13px; color: #c44; }
.result-card-s .label { font-size: 13px; color: #2a6e3f; margin: 0 0 4px 0; }
.result-card-s .value { font-size: 28px; font-weight: 700; color: #1a4d2e; margin: 0; }
.result-card-s .unit  { font-size: 13px; color: #3a8a55; }
.warn-box {
 background: #fff3cd;
 border-left: 4px solid #ffc107;
 border-radius: 6px;
 padding: 10px 14px;
 font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
 '<h1 style="font-size:96px !important; font-weight:900 !important; line-height:1.1; color:var(--text-color);">'
 'Hybrid Molecular Representation Combining Variational '
 'Autoencoder Latent Space and Physicochemical Descriptors for the Prediction '
 'of Standard Enthalpy of Formation and Standard Entropy</h1>',
 unsafe_allow_html=True,
)
st.markdown(
 '<p class="page-sub">Standard enthalpy of formation <b>Δ<sub>f</sub>H°</b> (kJ/mol) · '
 'Standard entropy <b>S°</b> (J/mol·K) · T = 298.15 K</p>',
 unsafe_allow_html=True,
)
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
 st.image("pipeline.png", caption="Schematic overview of the hybrid molecular representation pipeline", use_container_width=True)

with st.expander("Abstract", expanded=True):
 st.markdown("""
We introduce a hybrid molecular representation framework that couples **Variational Autoencoder (VAE)** latent vectors with physicochemical descriptors computed with the **RDKit** cheminformatics library to predict two fundamental gas-phase thermodynamic properties: the standard enthalpy of formation and the standard entropy.

The VAE is trained on a corpus of **53 895 SMILES strings** under a purely reconstructive objective, yielding a **128-dimensional latent embedding** that encodes molecular structure without any exposure to thermodynamic labels. This learned representation is concatenated with a curated set of physicochemical descriptors to form a **hybrid molecular fingerprint**, which is then supplied as input to a feedforward artificial neural network for property prediction.

The proposed model achieves:
- **R² = 0.9993** and **MAE = 5.98 kJ·mol⁻¹** for the standard enthalpy of formation
- **R² = 0.9955** and **MAE = 3.70 J·mol⁻¹·K⁻¹** for the standard entropy

An ablation study establishes that neither the VAE latent space nor the RDKit descriptors alone reach the accuracy of the combined fingerprint, confirming their genuine complementarity. A SHAP-based analysis reveals that the VAE latent features account for **25%** and **39%** of the one hundred most influential features for enthalpy and entropy, respectively. The proposed framework is **general**, **geometry-free**, and readily extensible to other molecular properties.
""")

st.markdown("---")

H_SKLEARN = "pipeline_enthalpy_saved/pipeline_enthalpy_sklearn.pkl"
H_ANN  = "pipeline_enthalpy_saved/pipeline_enthalpy_ann.pt"
H_CFG  = "pipeline_enthalpy_saved/pipeline_enthalpy_vae_config.pkl"
S_SKLEARN = "pipeline_entropy_saved/pipeline_entropy_sklearn.pkl"
S_ANN  = "pipeline_entropy_saved/pipeline_entropy_ann.pt"
S_CFG  = "pipeline_entropy_saved/pipeline_entropy_vae_config.pkl"
VAE_W  = "pretrained_VAE.pt"

if "pipelines" not in st.session_state:
 with st.spinner("Loading models, please wait..."):
  try:
   ann_H, sc_H, dp_H, vae_H, vocab_H, dn_H, dev = load_pipeline(
    H_SKLEARN, H_ANN, H_CFG, VAE_W)
   ann_S, sc_S, dp_S, vae_S, vocab_S, dn_S, _ = load_pipeline(
    S_SKLEARN, S_ANN, S_CFG, VAE_W)
   st.session_state["pipelines"] = {
    "H": (ann_H, sc_H, dp_H, vae_H, vocab_H, dn_H, dev),
    "S": (ann_S, sc_S, dp_S, vae_S, vocab_S, dn_S, dev),
   }
  except Exception as e:
   st.error(f"Model loading error: {e}")

models_loaded = "pipelines" in st.session_state

st.subheader("Molecule Input")

if "ketcher_confirmed" not in st.session_state:
 st.session_state["ketcher_confirmed"] = ""

mode = st.radio(
 "Input method",
 ["SMILES", "Draw molecule"],
 horizontal=True,
 label_visibility="collapsed",
)

smiles_list_to_predict = []

if mode == "SMILES":
 multi_raw = st.text_area(
  "Enter one SMILES per line",
  placeholder="CCO\nc1ccccc1\nCC(=O)O\n...",
  height=180,
  help="Paste or type one SMILES per line. Empty lines are ignored.",
  key="smiles_multi",
 )
 lines = [l.strip() for l in multi_raw.splitlines() if l.strip()]
 if lines:
  valid, invalid = [], []
  for l in lines:
   (valid if canonicalize(l) else invalid).append(l)
  st.caption(f"{len(valid)} valid SMILES · {len(invalid)} invalid")
  if invalid:
   st.warning("Invalid SMILES (will be skipped): " + ", ".join(f"`{s}`" for s in invalid))
  smiles_list_to_predict = valid

else:
 if not KETCHER_OK:
  st.warning(
   "The molecule editor requires `streamlit-ketcher`. "
   "Add it to requirements.txt and redeploy."
  )
 else:
  st.caption("Draw your molecule below, then click **Apply** in the editor.")
  ketcher_smi = st_ketcher(st.session_state["ketcher_confirmed"], height=420)
  if ketcher_smi and ketcher_smi.strip() != st.session_state["ketcher_confirmed"]:
   st.session_state["ketcher_confirmed"] = ketcher_smi.strip()
  confirmed = st.session_state["ketcher_confirmed"]
  if confirmed:
   canon = canonicalize(confirmed)
   if canon:
    st.caption(f"Canonical SMILES: `{canon}`")
    smiles_list_to_predict = [confirmed]
   else:
    st.warning("Could not parse the drawn molecule.")
predict_btn = st.button("Predict", type="primary")

if predict_btn:
 if not smiles_list_to_predict:
  st.warning("Please enter or draw a molecule first.")
 elif not models_loaded:
  st.error("Models are not loaded. Check that all model files are present in the repository.")
 else:
  pipes = st.session_state["pipelines"]
  with st.spinner("Computing prediction..."):
   df_H, errs_H = predict_smiles(smiles_list_to_predict, *pipes["H"], conv=ADIM_TO_KJMOL)
   df_S, errs_S = predict_smiles(smiles_list_to_predict, *pipes["S"], conv=R_GAS_CONSTANT)

  if errs_H or errs_S:
   st.markdown(
    '<div class="warn-box">Could not compute descriptors for some molecules. '
    'Check your SMILES input.</div>', unsafe_allow_html=True)

  if not df_H.empty or not df_S.empty:
   st.markdown("---")
   st.subheader("Prediction Results")

   result_df = pd.DataFrame({
    "SMILES (input)"  : df_H["SMILES (entré)"].values if not df_H.empty else df_S["SMILES (entré)"].values,
    "Canonical SMILES" : df_H["SMILES canonique"].values if not df_H.empty else df_S["SMILES canonique"].values,
    "ΔfH° (kJ/mol)"  : df_H["Valeur prédite"].values if not df_H.empty else [None]*len(df_S),
    "S° (J/mol·K)"   : df_S["Valeur prédite"].values if not df_S.empty else [None]*len(df_H),
   })

   for i, row in result_df.iterrows():
    smi_can = row["Canonical SMILES"]
    st.markdown(f"**Molecule {i+1}** — `{smi_can}`")
    c_img, c_h, c_s = st.columns([1.2, 1, 1])
    with c_img:
     img = mol_to_image(smi_can)
     if img:
      st.image(img, caption=smi_can, use_container_width=True)
    with c_h:
     val_H = row["ΔfH° (kJ/mol)"]
     if val_H is not None:
      st.markdown(
       f'<div class="result-card-h">'
       f'<p class="label">Standard Enthalpy of Formation</p>'
       f'<p class="value">{val_H:.2f}</p>'
       f'<p class="unit">kJ/mol &nbsp;·&nbsp; Δ<sub>f</sub>H°</p>'
       f'</div>', unsafe_allow_html=True)
    with c_s:
     val_S = row["S° (J/mol·K)"]
     if val_S is not None:
      st.markdown(
       f'<div class="result-card-s">'
       f'<p class="label">Standard Entropy</p>'
       f'<p class="value">{val_S:.2f}</p>'
       f'<p class="unit">J/mol·K &nbsp;·&nbsp; S°</p>'
       f'</div>', unsafe_allow_html=True)
    st.markdown("---")

   if len(result_df) > 1:
    st.subheader("Summary Table")
    st.dataframe(result_df, use_container_width=True)

   buf = io.BytesIO()
   result_df.to_excel(buf, index=False, engine="openpyxl")
   st.download_button(
    label="Download results (Excel)",
    data=buf.getvalue(),
    file_name="thermodynamic_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
   )
  elif not errs_H and not errs_S:
   st.error("No result obtained. Please verify the SMILES.")

st.markdown("---")
