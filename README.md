# DermaScan AI (CNN + Streamlit + AI assistant)

Pages: Home (`app.py`), Classifier, Disease Library, AI Assistant, Model Performance, About.

Files to upload to GitHub: everything in this folder plus `skin_cancer_cnn.keras` and
`class_names.json` (from Kaggle output). Optional: `assets/loss_curve.png`,
`assets/confusion_matrix.png`, `assets/metrics.json` (saved by the notebook).

Run locally: `pip install -r requirements.txt && streamlit run app.py`

AI assistant: on Streamlit Cloud open App -> Settings -> Secrets and add
`ANTHROPIC_API_KEY = "sk-ant-..."`. Without it the assistant falls back to the built-in knowledge base.
