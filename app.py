import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
@st.cache
def load_data():
    df = pd.read_csv('heloc_dataset_v1.csv')
    label_map = {'Good': 0, 'Bad': 1}
    df['RiskPerformance'] = df['RiskPerformance'].map(label_map)
    return df

df = load_data()

# 数据预处理和模型训练
def preprocess_and_train(df):
    X = df.iloc[:, 1:]
    Y = df['RiskPerformance']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

    # Preprocessing
    X_train = X_train[X_train['ExternalRiskEstimate'] != -9]
    Y_train = Y_train.loc[X_train.index]
    X_test = X_test[X_test['ExternalRiskEstimate'] != -9]
    Y_test = Y_test.loc[X_test.index]

    column_transformer = ColumnTransformer([
        ("imputer-7", SimpleImputer(missing_values=-7, strategy='mean'), X_train.columns),
        ("imputer-8", SimpleImputer(missing_values=-8, strategy='mean'), X_train.columns)
    ], remainder='passthrough')

    feature_expansion = FeatureUnion([
        ("columns", column_transformer),
        ("missing_indicator-7", MissingIndicator(missing_values=-7)),
        ("missing_indicator-8", MissingIndicator(missing_values=-8))
    ])

    pipeline = Pipeline([
        ("feature_expansion", feature_expansion),
        ("knn", KNeighborsClassifier(n_neighbors=5))  # 使用固定邻居数5进行示例
    ])

    pipeline.fit(X_train, Y_train)
    return pipeline, X_test, Y_test

model, X_test, Y_test = preprocess_and_train(df)

# 用户界面
st.title('HELOC Application Risk Evaluation')
if st.button('Predict Test Set'):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    cm = confusion_matrix(Y_test, y_pred)
    
    st.write(f"Test set accuracy: {accuracy:.2f}%")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)
