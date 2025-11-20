# Plant Disease Prediction with CNN

### Kaggle Dataset : https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

This repository contains a Jupyter notebook and a small Streamlit app that demonstrates building, training, evaluating, and deploying a Convolutional Neural Network (CNN) to classify plant diseases using the PlantVillage dataset.

---

## Contents

- `plant_disease_prediction.ipynb` — Full notebook: EDA, preprocessing, model building, training, evaluation, inference, and export.
- `app/` — Streamlit demo application and saved model:
  - `app/main.py` — Streamlit app for single-image inference.
  - `app/trained_model/` — Saved model files (HDF5 and/or SavedModel).
  - `app/requirements.txt` — App-specific Python dependencies.
- `class_indices.json` — Mapping from class indices to class names (used by the app).
- `kaggle.json` — (Not included) Your Kaggle credentials used to download datasets.

---

## Quick Setup

1. Create and activate a Python environment (recommended):

```bash
# Example (Windows PowerShell / cmd):
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# cmd
.\.venv\Scripts\activate.bat
```

2. Install the project dependencies. There are two possible requirement files:

- Global / notebook-level requirements (if present) — install at repo root
- App-specific dependencies:

```bash
pip install -r app/requirements.txt
# or, if you only have a single requirements file at root
# pip install -r requirements.txt
```

3. (Optional) If you need TensorFlow CPU/GPU specific wheels, install as needed. The notebook expects a working TensorFlow installation compatible with your Python version.

---

## Run the Notebook

Open `plant_disease_prediction.ipynb` in Jupyter, VS Code, or Colab and run cells sequentially. Important prerequisites:

- Ensure dataset files are available under the path referenced in the notebook (default: `plantvillage dataset/`).
- Confirm variables such as `train_paths`, `train_labels`, `class_names`, and `SEED` are defined before running pipeline/model cells.

To start Jupyter locally:

```bash
python -m notebook plant_disease_prediction.ipynb
# or
jupyter lab
```

---

## Run the Streamlit App (Demo)

Start the Streamlit app from the repository root. If the `streamlit` CLI is not on your PATH, run it via the Python module:

```bash
# Preferred (explicit Python executable)
C:/Python313/python.exe -m streamlit run app/main.py

# Or, if streamlit is installed in your active venv
streamlit run app/main.py
```

Open the Local URL printed by Streamlit (usually http://localhost:8501) to access the UI.

<img width="1913" height="953" alt="image" src="https://github.com/user-attachments/assets/d6277ccb-9ed0-4490-9201-8e6750f16898" />

---

<img width="1918" height="1002" alt="image" src="https://github.com/user-attachments/assets/7faa47e0-d569-4194-aeb2-3fe7fa42a8a1" />

---

<img width="1912" height="1013" alt="image" src="https://github.com/user-attachments/assets/ffdecf70-3094-4123-be68-908b50e8b830" />

---

## Retrain / Replace Model

If you want to retrain the model inside the notebook:

- Run the preprocessing cells to build `train_ds`, `val_ds`, and `test_ds`.
- Update model architecture (the notebook includes a simple CNN; for better accuracy prefer transfer learning).
- Train with callbacks (EarlyStopping, ModelCheckpoint) as shown in the notebook.
- Save the final model (both HDF5 and SavedModel formats are supported):

```python
model.save('app/trained_model/plant_disease_prediction_model.h5')
model.save('app/trained_model/plant_disease_prediction_savedmodel')
```

---

## Export & Deployment Notes

- For serving with TensorFlow Serving, use the SavedModel format.
- For mobile/embedded, convert to TFLite after testing accuracy.
- For a production REST API, wrap the model with FastAPI/Flask and containerize with Docker (a `Dockerfile` exists in `app/`).

---

## Troubleshooting

- `streamlit: command not found` — Install Streamlit in the environment and run using `python -m streamlit run app/main.py` if the CLI is not on PATH.
- Model load errors — Verify `app/trained_model/plant_disease_prediction_model.h5` exists and is compatible with installed TF version.
- Dataset path errors — Confirm `plantvillage dataset/` path and that the notebook's `base_path` matches your layout.

---

## Next Steps & Improvements

- Replace the small CNN with a transfer-learning backbone (e.g., EfficientNet, MobileNetV2) for improved accuracy.
- Add automated training logs (TensorBoard) and model versioning.
- Add unit tests for preprocessing and prediction helpers.
- Improve the Streamlit frontend with more explanatory UI and an option to save predictions.

---

## Contact

If you need help running the project or want me to implement any of the suggested improvements, reply here or open an issue.

---


