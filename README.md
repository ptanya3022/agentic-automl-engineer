# 🚀 Agentic AutoML Engineer

### 🤖 A powerful agent-based AutoML assistant that automates the entire ML pipeline — from preprocessing to model selection, hyperparameter tuning, and performance reporting — all wrapped in an intuitive Streamlit interface.

🔗 [![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-blueviolet?style=for-the-badge&logo=streamlit)](https://agentic-automl-engineer-poemderjyy3jkj3qputpxz.streamlit.app/)



📁 **Dataset Sample**: `diabetes.csv` (or any tabular dataset)

---

## 📌 Tech Stack

- **Python 3.11+**
- **Streamlit**
- **Scikit-learn**
- **XGBoost**
- **Optuna** (for hyperparameter tuning)
- **Pandas & NumPy**
  
---

## ✨ Project Highlights

- ⚙️ **Built a fully automated machine learning pipeline** that selects and tunes the best model from Logistic Regression, Random Forest, and XGBoost based on dataset characteristics.
- 📈 **Achieved up to 90%+ accuracy** on classification problems using optimized pipelines with minimal user intervention.
- 💡 **Empowered users to run AutoML experiments** without writing a single line of code — democratizing access to ML insights.

---

## 🗂️ Project Structure

```
Agentic-AutoML-Engineer/
│
├── app.py                  # Streamlit front-end
├── pipeline.py             # Core AutoML logic
├── requirements.txt        # All dependencies
├── output/
│   ├── classification_report.csv  # Saved report after each run
├── .gitignore
└── README.md
```

---

## 🧠 How It Works

1. **Upload a CSV Dataset**
2. **Select the target column**
3. **Click "Run AutoML Pipeline"**
4. **Get model performance, accuracy, and classification report**
5. **Download the result from the output folder**

---

## 📈 Business Impact

> Enables domain experts, analysts, and beginners to run fast AutoML experiments on their own data without writing code — making it ideal for **industry rapid prototyping**, **proof-of-concepts**, or **internal tooling**.

---

## 🔮 Future Scope

- Extend to support **regression problems**
- Add support for **multi-class datasets**
- Integrate with **LangGraph** to allow natural language-driven pipeline creation
- Add **cloud storage** integration for persistent outputs

---

## 🙏 Acknowledgements

- Streamlit for rapid UI prototyping
- Scikit-learn, Optuna, and XGBoost — the real MVPs

---

## 🚀 Getting Started Locally

```bash
# Clone the repo
git clone https://github.com/ptanya3022/agentic-autoML-engineer.git
cd agentic-autoML-engineer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🪪 License

This project is intended for educational and non-commercial use only.
