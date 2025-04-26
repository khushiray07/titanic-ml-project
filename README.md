# titanic-ml-project
# Titanic - Machine Learning from Disaster 🚢

This project predicts the survival of passengers on the Titanic using machine learning models.  
The dataset is taken from the famous Kaggle Titanic competition.

---

## 📚 Project Overview

- **Goal:** Predict whether a passenger survived the Titanic disaster.
- **Models Used:** Logistic Regression, Random Forest Classifier
- **Final Accuracy:** ~82.68% (Random Forest)

---

## 🛠️ Technologies & Libraries Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📊 Workflow

1. **Data Loading**
   - Loaded Titanic dataset (`train.csv`) using Pandas.

2. **Exploratory Data Analysis (EDA)**
   - Visualized survival distribution, gender-wise survival, class-wise survival.

3. **Data Cleaning**
   - Filled missing Age values with median.
   - Filled missing Embarked values with mode.
   - Dropped irrelevant columns (Name, Ticket, PassengerId, Cabin).

4. **Feature Engineering**
   - Converted categorical variables (Sex, Embarked) into numerical values.

5. **Model Building**
   - Trained Logistic Regression and Random Forest models.
   - Used 80% of data for training and 20% for testing.

6. **Model Evaluation**
   - Accuracy, Confusion Matrix, Precision, Recall, F1-Score.
   - Random Forest achieved the best accuracy (~82.68%).

7. **Visualization**
   - Confusion matrix plotted using Seaborn heatmap for Random Forest predictions.

---

## 📈 Results

| Model | Accuracy |
|:-----|:---------|
| Logistic Regression | ~79.88% |
| Random Forest Classifier | ~82.68% |

✅ Random Forest p



---

## 📢 Future Improvements

- Hyperparameter tuning for Random Forest (using GridSearchCV).
- Feature engineering: create age groups, family size features.
- Try advanced models like XGBoost, LightGBM.

---

## 🙋‍♂️ Author

- **Name:** [Khushi Ray]
- **GitHub:** [https://github.com/khushiray07](https://github.com/khushiray07)
- **LinkedIn:** [https://www.linkedin.com/in/ray-khushi/](https://www.linkedin.com/in/ray-khushi/)

---

## 🎯 Important Note

This project is for educational purposes (learning machine learning workflows).  
The dataset comes from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).

---
