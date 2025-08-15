### App Link : https://insurance-cost-prediction-lalitsonii240.streamlit.app/


### 1. Outline of the Problem Statement, Target Metric, and Steps Taken

#### Problem Statement
The project focuses on predicting individual health insurance costs (premium prices) using machine learning to enable insurance companies to set more accurate, personalized premiums. Traditional methods rely on broad actuarial tables and averages, which fail to account for individual nuances like health conditions, age, and lifestyle factors. This leads to inaccurate pricing, potential financial losses for insurers, unfairly high premiums for customers, and reduced competitiveness. The model aims to enhance pricing precision, improve risk assessment, boost customer satisfaction through fair and transparent pricing, enable personalized insurance offerings, and inform strategic decisions like policy development and market entry.

The dataset used is `insurance.csv`, containing 986 records with 11 features: Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries, and the target variable PremiumPrice.

#### Target Metric
The primary target is **PremiumPrice** (a continuous variable representing the insurance premium cost in currency units, ranging from 15,000 to 40,000). As this is a regression problem, the key evaluation metrics are:
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors (lower is better; prioritizes larger errors).
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values (lower is better; easier to interpret in the context of premium costs).
- **R² Score**: Indicates the proportion of variance in PremiumPrice explained by the model (higher is better; closer to 1 means better fit).

These metrics were used across cross-validation (CV) and test sets to assess model performance, with RMSE as the primary scoring metric for hyperparameter tuning.

#### Steps Taken to Solve the Problem
The project follows a structured pipeline: Data Loading → Exploratory Data Analysis (EDA) → Feature Engineering → Hypothesis Testing → Machine Learning Modeling. Insights and recommendations are derived at each stage.

##### a. Exploratory Data Analysis (EDA)
- **Data Inspection**: Loaded data from Google Drive. Dataset shape: (986 rows, 11 columns). No missing values or duplicates. All columns are integers (binary for categorical features like Diabetes; continuous for Age, Height, Weight, PremiumPrice).
- **Summary Statistics**: 
  - Categorical (binary/multi-class): Diabetes (42% yes), BloodPressureProblems (47% yes), AnyTransplants (6% yes), AnyChronicDiseases (18% yes), KnownAllergies (22% yes), HistoryOfCancerInFamily (12% yes), NumberOfMajorSurgeries (0-3, mostly 0-1).
  - Numerical: Age (mean 41.7, 18-66), Height (mean 168 cm), Weight (mean 77 kg), PremiumPrice (mean 24,337).
- **Visualizations**:
  - Univariate: Histograms/KDEs showed uniform Age distribution, normal-ish Height/Weight with right skew, right-skewed PremiumPrice (peak at 24,000).
  - Bivariate: Boxplots revealed higher premiums for those with Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, HistoryOfCancerInFamily, and more surgeries. Scatterplots showed positive correlations between Age/Weight and PremiumPrice; negligible for Height.
- **Correlation Analysis**: Heatmap showed strong positive correlations of PremiumPrice with Age (0.70), AnyTransplants (0.29), NumberOfMajorSurgeries (0.26); moderate with AnyChronicDiseases (0.21), BloodPressureProblems (0.17).
- **Outlier Detection**: Used IQR on continuous variables (e.g., Weight >118 kg, PremiumPrice >39,000). Retained outliers as they represent valid high-risk cases in a small dataset (986 rows), preserving predictive signals for robust models like trees.
- **Insights**: Age and health conditions (e.g., transplants, chronic diseases) are key drivers. Dataset is balanced for most binaries but imbalanced for rare events (e.g., transplants). No multicollinearity (VIF <1.4 for all features).
- **Recommendations**: Proceed with all features; monitor imbalances in modeling (e.g., via class weighting if needed, though not critical for regression).

##### b. Feature Engineering
- Created BMI from Height/Weight: BMI = Weight / (Height/100)^2, dropped Height/Weight to avoid redundancy.
- Scaled numerical features (Age, BMI) using StandardScaler for scale-sensitive models (e.g., Linear Regression, Neural Networks).
- Insights: BMI showed right-skewed distribution (peak 25-28); slight positive correlation with PremiumPrice. However, adding BMI and scaling slightly decreased performance in tree-based models (as they are scale-invariant) but helped in linear models.
- Recommendations: Use original features without BMI/scaling for tree models; include for others. Avoid over-engineering in small datasets to prevent overfitting.

##### c. Hypothesis Testing
- **Parametric Tests (for PremiumPrice means across groups)**:
  - T-tests: Significant differences for Diabetes (p=0.0145), BloodPressureProblems (p<0.0001), AnyTransplants (p<0.0001), AnyChronicDiseases (p<0.0001), HistoryOfCancerInFamily (p=0.0198); insignificant for KnownAllergies (p=0.7141).
  - ANOVA: Significant for NumberOfMajorSurgeries (p<0.0001).
- **Chi-Square (associations between categoricals)**: Significant for Diabetes-AnyChronicDiseases (p=0.0064), KnownAllergies-HistoryOfCancerInFamily (p=0.0005), Diabetes-BloodPressureProblems (p=0.0001); insignificant for others (e.g., Diabetes-AnyTransplants p=0.3123).
- **Regression Analysis**: OLS model (R²=0.7134) confirmed significant predictors: Age, AnyTransplants, AnyChronicDiseases, Weight, HistoryOfCancerInFamily, NumberOfMajorSurgeries (all p<0.001); insignificant: Diabetes, BloodPressureProblems, Height, KnownAllergies.
- Insights: Confirms EDA findings—Age and major health events strongly influence premiums. Associations like Diabetes-BloodPressureProblems suggest clustered risks.
- Recommendations: Drop insignificant features (e.g., Height, KnownAllergies) in final models to reduce noise. Use insights for feature selection (prioritize Age, transplants).

##### d. Machine Learning Modeling
- **Data Split**: 80/20 train-test; 5-fold CV for tuning.
- **Models Trained** (with GridSearchCV/RandomizedSearchCV for hyperparameters like max_depth, n_estimators, learning_rate):
  - Linear Regression: Simple baseline; tuned fit_intercept, positive.
  - Decision Tree: Tuned max_depth, min_samples_split/leaf, max_features.
  - Random Forest: Ensemble of trees; tuned n_estimators, max_depth, min_samples_split.
  - Gradient Boosting: Sequential boosting; tuned n_estimators, learning_rate, max_depth, subsample.
  - Neural Network: Keras-based; tuned learning_rate, architecture (64-32-1 neurons, dropout 0.2, Adam optimizer, 50 epochs).
- **Feature Importance/Permutation Importance**: Across models, Age dominated (60-77% importance), followed by AnyTransplants (10-11%), Weight (5-10%), AnyChronicDiseases (3-4%). Confidence intervals via bootstrapping confirmed stability.
- **Insights**: Tree-based models outperformed linear/NN due to non-linear relationships (e.g., interactions between Age and health conditions). Overfitting minimal (test RMSE close to CV). BMI/scaling hurt tree models but helped NN/Linear.
- **Recommendations**: Select Random Forest as best (balances accuracy and interpretability). Ensemble methods handle outliers/imbalances well. For production, monitor for concept drift (e.g., changing health trends).

#### Overall Insights and Recommendations
- **Insights**: The problem is driven by age and severe health factors; minor conditions (e.g., allergies) have low impact. Models capture 71-90% variance, indicating good predictability but room for more data/features (e.g., lifestyle, income).
- **Recommendations**: 
  - Collect more data for rare events (e.g., transplants) to improve minority predictions.
  - Use SHAP/XAI for interpretability in production.
  - Retrain periodically with new data.
  - Ethical considerations: Ensure model avoids bias (e.g., against older/chronic patients) via fairness audits.

### 2. Final Scores Achieved and Deployment Steps

#### Final Scores Achieved
Multiple models were compared on the test set (20% holdout). Random Forest emerged as the best overall, with the lowest RMSE/MAE and highest R², showing superior handling of non-linearities and interactions. Feature engineering (BMI) and scaling were tested but discarded for tree models as they reduced performance (e.g., increased RMSE by 10-20%). Final scores (without BMI/scaling for trees; with for Linear/NN):

| Model              | CV RMSE | Test RMSE | Test MAE | Test R² | Best Hyperparameters |
|--------------------|---------|-----------|----------|---------|----------------------|
| **Linear Regression** | N/A    | 3495.95  | 2586.23 | 0.7134 | fit_intercept=True, positive=False, n_jobs=-1 |
| **Decision Tree**    | 3034.22| 2177.71  | 1044.65 | 0.8888 | max_depth=7, max_features=None, min_samples_leaf=1, min_samples_split=20 |
| **Random Forest** (Best) | 2979.44| 2071.14  | 992.53  | 0.8994 | max_depth=15, min_samples_split=10, n_estimators=200 |
| **Gradient Boosting**| 3075.41| 2386.98  | 1551.38 | 0.8664 | learning_rate=0.05, max_depth=3, min_samples_split=2, n_estimators=100, subsample=0.8 |
| **Neural Network**   | N/A    | ~3500-3600 (approx., from code output) | ~2600-2700 | ~0.70-0.71 | learning_rate=0.01 (or 0.001), epochs=50, batch_size=64 |

- **Interpretation**: Random Forest's Test RMSE (2071) means average prediction error is ~₹2071, acceptable for premiums averaging ₹24,337. High R² (0.90) indicates excellent fit. Overfitting check: Test scores close to CV, no major gaps.
- **Model Selection Rationale**: Random Forest for production—robust, interpretable via feature importance, handles small data well.

#### Steps Taken for Deployment
Deployment uses Streamlit for a web app and pyngrok for tunneling (local to public URL). The final model (Random Forest) is saved via joblib. Steps from the PDF/code:

1. **Model Saving**: After training, save the best model and scaler (if used): `joblib.dump(best_model, 'best_random_forest_model.pkl')`; `joblib.dump(scaler, 'scaler.pkl')` (though scaling skipped for final tree model).

2. **Streamlit App Setup**:
   - Install: `!pip install streamlit pyngrok`.
   - Create `app.py`: Import libraries (pandas, joblib, streamlit). Load model/scaler.
   - UI: Use `st.title('Insurance Premium Predictor')`; inputs via sliders/selectboxes for features (e.g., `age = st.slider('Age', 18, 66)`).
   - Prediction: On button click, preprocess inputs (e.g., create DataFrame, scale if needed), predict: `prediction = model.predict(input_df)`; display: `st.write(f'Predicted Premium: ₹{prediction[0]:.2f}')`.

3. **Local Run**: `streamlit run app.py` (runs on localhost:8501).

4. **Public Exposure with pyngrok**:
   - Get ngrok auth token (from ngrok.com).
   - In code: `from pyngrok import ngrok`; `public_url = ngrok.connect(8501)`; print public URL.
   - Access app via generated URL (e.g., for demos/sharing).

5. **Testing**: Input sample data (e.g., from df.head()), verify predictions match model output. Handle errors (e.g., invalid inputs).

6. **Additional Notes**: App includes feature importance visualization (e.g., bar plot). For production, deploy on Heroku/AWS; add authentication. No cloud API mentioned, but xAI API redirect if queried.

Insights: Deployment is beginner-friendly for portfolios; scalable for real use with Docker/Kubernetes. Recommendations: Add data validation, logging, and A/B testing for model updates.
