# 📊 Sales Prediction Model using Machine Learning

## 📝 Project Overview
This project builds a **Sales Prediction Model** using **Machine Learning** to analyze and predict sales based on historical transaction data. The dataset contains **9,994 transactions** with attributes like customer details, product sales, discounts, and profits. The model helps businesses forecast future sales trends, optimize pricing, and improve decision-making.

## 📂 Dataset Details
The dataset is an Amazon AWS SaaS Sales dataset containing transaction details. Key features include:

- **Order Date:** Date of transaction
- **Country, City, Region:** Location details of customers
- **Industry & Segment:** Business classification
- **Product, Quantity, Discount, Profit:** Transaction details
- **Sales (Target Variable):** Total sales per transaction

✅ **No missing values**  
✅ **Data types verified**

## 🛠️ Tech Stack & Libraries
- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn (Linear Regression, Random Forest, Gradient Boosting)
- Jupyter Notebook

## 🚀 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/sales-prediction-ml.git
cd sales-prediction-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Exploratory Data Analysis (EDA)
Performed data visualization & outlier detection:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sales Distribution
sns.histplot(df['Sales'], bins=30, kde=True)
plt.title('Sales Distribution')
plt.show()
```

## 🔍 Data Preprocessing & Feature Engineering
- Removed **irrelevant columns** (Order ID, Contact Name, License, etc.)
- **Handled outliers** using IQR method
- **Encoded categorical variables** using One-Hot Encoding
- **Scaled numerical features** using StandardScaler

## 🤖 Model Training & Evaluation
We tested multiple models to find the best-performing one:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
```

### ✅ Best Model: **Random Forest Regressor**
📌 **MAE:** _Reduced significantly_  
📌 **RMSE:** _Better accuracy compared to Linear Regression_  
📌 **R² Score:** _Closer to 1 (better performance)_

## 📌 Results & Insights
- The **Random Forest Model** outperformed other models.
- Discounts and Quantity had a strong impact on Sales.
- The **Sales distribution showed some skewness**, requiring log transformation for improvement.

## 📈 Future Improvements
- Implement **Hyperparameter tuning** for better accuracy.
- Try **Deep Learning models** for improved forecasting.
- Deploy the model using **Flask or FastAPI** for real-time predictions.

## 🤝 Contributing
Feel free to contribute to this project by opening a pull request! 🚀

## 👨‍💻 Author
**Tejas Krishna A S**  
🚀 AI/ML Engineer | Data Analyst 


