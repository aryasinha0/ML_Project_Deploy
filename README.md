# Machine Learning Prediction Web App

A machine learning web application that provides predictions through a user-friendly Flask interface. The trained ML model is integrated into a Flask backend and deployed as a web service using Render.

## 🚀 Live Demo

**Live Application:** [https://ml-project-deploy-k2f4.onrender.com/]

## 📌 Project Overview

This project demonstrates the complete workflow of taking a machine learning model from development to deployment.

The application:

* Accepts user input through a web interface
* Preprocesses the input data
* Loads a pre-trained machine learning model
* Generates predictions
* Displays the prediction to the user
* Provides a deployable web interface using Flask

## 🛠️ Technologies Used

* **Python**
* **Flask** – Web application framework
* **Scikit-learn** – Machine learning
* **Pandas** – Data preprocessing and manipulation
* **NumPy** – Numerical computations
* **Joblib** – Model serialization
* **HTML/CSS** – Frontend
* **Gunicorn** – Production WSGI server
* **Render** – Deployment platform

## 📂 Project Structure

```text
ML_Project/
│
├── app.py                  # Flask application
├── model.pkl               # Trained ML model
├── preprocessing.pkl       # Preprocessing object(s), if applicable
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web interface
├── static/
│   ├── style.css           # Styling
│   
├── .gitignore
└── README.md
```

## ⚙️ How It Works

```text
User Input
    ↓
Flask Web Application
    ↓
Input Validation & Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Result Displayed on Web Page
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/aryasinha0/ML_Project_Deploy.git
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

## ☁️ Deployment on Render

The application can be deployed on Render as a Web Service.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

After deployment, Render provides a public URL that can be used to access the application.

## 🧠 Machine Learning Model

The application uses a pre-trained machine learning model that was developed and evaluated before deployment.

The trained model is serialized using Joblib and loaded by the Flask application when the server starts.

Example:

```python
import joblib

model = joblib.load("model.pkl")
```

If preprocessing is performed separately, the preprocessing object is also loaded before making predictions.

## 🔄 Prediction Workflow

1. User enters the required features.
2. Flask receives the submitted data.
3. The application validates and preprocesses the input.
4. The pre-trained model generates a prediction.
5. The prediction is returned to the frontend.
6. The result is displayed to the user.


## 🔒 Notes

* The model is loaded from a serialized file and is **not retrained during application requests**.
* Model and preprocessing files must be available in the project directory when the application starts.
* Dependency versions should be kept compatible with the versions used to train the saved model.
* Sensitive information such as API keys and credentials should not be committed to the repository.

## 🔮 Future Improvements

* Add better input validation and error handling
* Improve the frontend user experience
* Add model explainability
* Add prediction history
* Containerize the application using Docker
* Add automated CI/CD
* Monitor model performance after deployment

## 👤 Author

**Arya Sinha**

---

⭐ If you found this project useful, consider giving the repository a star!
