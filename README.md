# 🏥 Medical Insurance Cost Predictor

ML Pipeline that predicts medical insurance charges based on user demographics and lifestyle inputs. Built with scikit-learn pipelines, deployed using Streamlit and Docker on Hugging Face Spaces.

## 🔗 Live Demo
👉 [Try the app on Hugging Face Spaces](https://huggingface.co/spaces/urvidhomne/Insurance-Cost-Predictor)

---

## 📌 Project Overview

This project estimates insurance costs based on input features such as age, BMI, smoker status, and region. It demonstrates an end-to-end ML pipeline — from preprocessing and modeling to deployment in a production environment.

### Features
- Real-time prediction of insurance charges
- Trained using Ridge and XGBoost regressors
- Integrated scikit-learn pipeline with ColumnTransformer & GridSearchCV
- Dockerized and deployed on Hugging Face Spaces
- Interactive web UI using Streamlit

---

## 🧪 Model Performance

- ✅ Best Model: **Ridge Regression**
- ✅ R² Score: **0.67**
- ✅ Evaluation Metrics: MAE = 0.22, RMSE = 0.44 (scaled)
