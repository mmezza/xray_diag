# Imagens de Teste — Raios-X Sintéticos

6 imagens sintéticas geradas para testar o app sem necessidade de dados reais.

| Arquivo | Cenário | Região |
|---------|---------|--------|
| `torax_normal_pa.png` | Raio-X normal, sem alterações | Tórax PA |
| `torax_pneumonia_lobo_inferior.png` | Consolidação em lobo inferior D | Tórax PA |
| `torax_derrame_pleural.png` | Opacidade em base E | Tórax PA |
| `torax_cardiomegalia.png` | Índice cardiotorácico aumentado | Tórax PA |
| `torax_nodulo_pulmonar.png` | Nódulo solitário em campo médio D | Tórax PA |
| `coluna_fratura_vertebral.png` | Colapso vertebral em T8 (lateral) | Coluna Torácica |

## Como usar

Execute `generate_samples.py` para regenerar as imagens:

```bash
pip install pillow numpy
python generate_samples.py
```

## Datasets reais recomendados

Para testes com imagens reais:

- **NIH ChestX-ray14** — https://nihcc.app.box.com/v/ChestXray-NIHCC
- **CheXpert (Stanford)** — https://stanfordmlgroup.github.io/competitions/chexpert/
- **RSNA Pneumonia Detection** — https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
- **MIMIC-CXR** — https://physionet.org/content/mimic-cxr/2.0.0/ (requer credenciais)
