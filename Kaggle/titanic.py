# -*- coding: utf-8 -*-
"""
Titanic Survival Prediction Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TitanicPredictor:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.best_model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load training and testing data"""
        print("=" * 50)
        print("1. Loading Data")
        print("=" * 50)
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')
        print(f"Train set size: {self.train_df.shape}")
        print(f"Test set size: {self.test_df.shape}")
        print("\nFirst 5 rows of training data:")
        print(self.train_df.head())

    def explore_data(self):
        """Exploratory Data Analysis"""
        print("\n" + "=" * 50)
        print("2. Data Exploration")
        print("=" * 50)

        print("\nBasic Info:")
        print(self.train_df.info())

        print("\nStatistical Summary:")
        print(self.train_df.describe())

        print("\nMissing Values:")
        missing_train = self.train_df.isnull().sum()
        missing_test = self.test_df.isnull().sum()
        print("Training set:")
        print(missing_train[missing_train > 0])
        print("\nTest set:")
        print(missing_test[missing_test > 0])

        print(f"\nOverall Survival Rate: {self.train_df['Survived'].mean():.2%}")

        print("\nSurvival by Features:")
        print(f"Gender: Female {self.train_df[self.train_df['Sex']=='female']['Survived'].mean():.2%}, "
              f"Male {self.train_df[self.train_df['Sex']=='male']['Survived'].mean():.2%}")
        print(f"Passenger Class: ")
        for pclass in [1, 2, 3]:
            survival_rate = self.train_df[self.train_df['Pclass']==pclass]['Survived'].mean()
            print(f"  Class {pclass}: {survival_rate:.2%}")

        print("\nFeature Correlations with Survival:")
        numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        correlation = self.train_df[numeric_features].corr()
        print(correlation['Survived'].sort_values(ascending=False))

    def feature_engineering(self):
        """Feature Engineering"""
        print("\n" + "=" * 50)
        print("3. Feature Engineering")
        print("=" * 50)

        all_data = pd.concat([self.train_df, self.test_df], sort=False).reset_index(drop=True)

        print("\nHandling missing values...")
        all_data['Age'].fillna(all_data['Age'].median(), inplace=True)
        all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
        all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
        all_data['HasCabin'] = all_data['Cabin'].notna().astype(int)

        print("Creating new features...")
        all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
        all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

        all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        all_data['Title'] = all_data['Title'].map(title_mapping)
        all_data['Title'].fillna('Rare', inplace=True)

        all_data['AgeGroup'] = pd.cut(all_data['Age'], bins=[0, 12, 18, 35, 60, 100],
                                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

        all_data['FareGroup'] = pd.qcut(all_data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

        print("Encoding categorical variables...")
        all_data['Sex'] = LabelEncoder().fit_transform(all_data['Sex'])
        all_data['Embarked'] = LabelEncoder().fit_transform(all_data['Embarked'])
        all_data['Title'] = LabelEncoder().fit_transform(all_data['Title'])
        all_data['AgeGroup'] = LabelEncoder().fit_transform(all_data['AgeGroup'])
        all_data['FareGroup'] = LabelEncoder().fit_transform(all_data['FareGroup'])

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                   'FamilySize', 'IsAlone', 'Title', 'HasCabin', 'AgeGroup', 'FareGroup']

        print(f"\nFinal features: {features}")

        train_len = len(self.train_df)
        self.train_df = all_data[:train_len]
        self.test_df = all_data[train_len:]

        self.X_train = self.train_df[features]
        self.y_train = self.train_df['Survived']

        print(f"Training feature dimensions: {self.X_train.shape}")

    def train_models(self):
        """Train and compare multiple models"""
        print("\n" + "=" * 50)
        print("4. Model Training & Comparison")
        print("=" * 50)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'SVM': SVC(kernel='rbf', random_state=42)
        }

        results = {}
        print("\nEvaluating models with 5-fold cross-validation:")
        print("-" * 50)

        for name, model in models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            results[name] = scores
            print(f"{name:25s} | Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        best_model_name = max(results, key=lambda x: results[x].mean())
        print(f"\nBest Model: {best_model_name}")

        print(f"\nTuning hyperparameters for {best_model_name}...")

        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
        elif best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = XGBClassifier(random_state=42, eval_metric='logloss')
        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingClassifier(random_state=42)
        else:
            self.best_model = models[best_model_name]
            self.best_model.fit(self.X_train, self.y_train)
            return

        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

        if hasattr(self.best_model, 'feature_importances_'):
            print("\nFeature Importance Ranking:")
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance)

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "=" * 50)
        print("5. Model Evaluation")
        print("=" * 50)

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

        self.best_model.fit(X_train_split, y_train_split)
        y_pred = self.best_model.predict(X_val_split)

        print(f"\nValidation Accuracy: {accuracy_score(y_val_split, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val_split, y_pred, target_names=['Not Survived', 'Survived']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val_split, y_pred))

        print("\nRetraining on full training data...")
        self.best_model.fit(self.X_train, self.y_train)

    def predict_and_submit(self):
        """Generate predictions and submission file"""
        print("\n" + "=" * 50)
        print("6. Generating Predictions")
        print("=" * 50)

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                   'FamilySize', 'IsAlone', 'Title', 'HasCabin', 'AgeGroup', 'FareGroup']
        X_test = self.test_df[features]

        predictions = self.best_model.predict(X_test)

        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'].astype(int),
            'Survived': predictions.astype(int)
        })

        submission.to_csv('submission.csv', index=False)
        print("\nPrediction completed!")
        print(f"Predicted survivors: {predictions.sum()}")
        print(f"Predicted survival rate: {predictions.mean():.2%}")
        print("\nSubmission file saved as: submission.csv")
        print("First 10 predictions:")
        print(submission.head(10))

    def run(self):
        """Run complete prediction pipeline"""
        self.load_data()
        self.explore_data()
        self.feature_engineering()
        self.train_models()
        self.evaluate_model()
        self.predict_and_submit()

        print("\n" + "=" * 50)
        print("All Done! 🎉")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Check submission.csv file")
        print("2. Login to Kaggle: https://www.kaggle.com/c/titanic")
        print("3. Upload submission.csv")
        print("4. Check your score and ranking!")


if __name__ == '__main__':
    predictor = TitanicPredictor()
    predictor.run()
