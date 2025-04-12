"""
Wine Sentiment Analysis DAG

This DAG processes wine review data, calculates subjectivity,
trains various sentiment analysis models, and saves the best model using MLflow.
"""

from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import joblib 

import mlflow
import mlflow.sklearn
import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'wine_sentiment_analysis',
    default_args=default_args,
    description='Wine reviews sentiment analysis and recommendation engine',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 4, 12),
    catchup=False,
    tags=['wine', 'sentiment_analysis', 'mlops'],
)

DATA_PATH = '/opt/airflow/data/wine_profiles6.csv'
MLFLOW_TRACKING_URI = "http://mlflow:5000" 

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://s3:9000"
EXPERIMENT_NAME = "wine-sentiment-analysis"

# Function to load data
def load_data(**kwargs):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully with shape: {df.shape}")
        
        required_cols = [
            'wine_id', 'variedad', 'sentiment_label', 'polarity', 
            'full_text', 'price', 'title'
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in dataset: {missing_cols}")
            for col in missing_cols:
                df[col] = None
                
        kwargs['ti'].xcom_push(key='wine_df', value=df.to_json(orient='split'))
        return "Data loaded successfully"
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Function to calculate subjectivity
def calculate_subjectivity(**kwargs):
    """Calculate subjectivity for each review using TextBlob."""
    ti = kwargs['ti']
    df_json = ti.xcom_pull(key='wine_df', task_ids='load_data')
    df = pd.read_json(df_json, orient='split')
    
    # Function to calculate subjectivity with TextBlob
    def calc_subjectivity(text):
        try:
            return TextBlob(str(text)).sentiment.subjectivity
        except:
            return 0.0
            
    # Apply the function to calculate subjectivity
    df['subjectivity'] = df['full_text'].apply(calc_subjectivity)
    
    # Map sentiment labels to numeric values
    label_map = {'Positivo': 1, 'Neutral': 0, 'Negativo': -1}
    df['y'] = df['sentiment_label'].map(label_map)
    
    print(f"Subjectivity calculated successfully. Sample: {df[['wine_id', 'title', 'subjectivity']].head()}")
    print(f"Class distribution: {df['sentiment_label'].value_counts()}")
    
    # Pass the updated DataFrame to the next task
    kwargs['ti'].xcom_push(key='preprocessed_df', value=df.to_json(orient='split'))
    return "Subjectivity calculated successfully"

# Function to vectorize text and prepare for model training
def prepare_features(**kwargs):
    """Vectorize text using TF-IDF and prepare features for model training."""
    ti = kwargs['ti']
    df_json = ti.xcom_pull(key='preprocessed_df', task_ids='calculate_subjectivity')
    df = pd.read_json(df_json, orient='split')
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['full_text'].fillna(''))
    y = df['y']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the vectorizer for later use
    joblib.dump(vectorizer, '/tmp/tfidf_vectorizer.joblib')
    
    # Convert sparse matrices to format that can be passed through XCom
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Pass the prepared data to the next task
    kwargs['ti'].xcom_push(key='X_train', value=X_train_dense.tolist())
    kwargs['ti'].xcom_push(key='X_test', value=X_test_dense.tolist())
    kwargs['ti'].xcom_push(key='y_train', value=y_train.tolist())
    kwargs['ti'].xcom_push(key='y_test', value=y_test.tolist())
    
    print(f"Features prepared with shape: X_train {X_train.shape}, X_test {X_test.shape}")
    return "Features prepared successfully"

# Function to train models
def train_models(**kwargs):
    """Train multiple sentiment analysis models and save the best one."""
    ti = kwargs['ti']
    
    # Get data from XCom
    X_train = np.array(ti.xcom_pull(key='X_train', task_ids='prepare_features'))
    X_test = np.array(ti.xcom_pull(key='X_test', task_ids='prepare_features'))
    y_train = np.array(ti.xcom_pull(key='y_train', task_ids='prepare_features'))
    y_test = np.array(ti.xcom_pull(key='y_test', task_ids='prepare_features'))
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Dictionary to store model results
    models_results = {}
    
    # Original Polarity-based method as baseline
    df_json = ti.xcom_pull(key='preprocessed_df', task_ids='calculate_subjectivity')
    df = pd.read_json(df_json, orient='split')
    
    # Function to map polarities to labels
    def polarity_to_label(p):
        if p > 0:
            return 1  # Positive
        elif p < 0:
            return -1  # Negative
        else:
            return 0  # Neutral
    
    # Get indices for test data
    df_indices = np.arange(len(df))
    _, X_test_indices, _, _ = train_test_split(
        df_indices, df['y'], test_size=0.2, random_state=42, stratify=df['y']
    )
    
    # Get true labels and predictions based on polarities
    y_pred_polarity = df.iloc[X_test_indices]['polarity'].apply(polarity_to_label)
    
    # Calculate metrics for polarity-based approach
    acc_polarity = accuracy_score(y_test, y_pred_polarity)
    f1_polarity = f1_score(y_test, y_pred_polarity, average='macro')
    print(f"Polarity method - Accuracy: {acc_polarity:.4f}, F1 Score: {f1_polarity:.4f}")
    models_results['polarity'] = {'accuracy': acc_polarity, 'f1_score': f1_polarity}
    
    # Train and evaluate Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression"):
        # Train model
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        
        # Predict
        y_pred_lr = lr_model.predict(X_test)
        
        # Calculate metrics
        acc_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr, average='macro')
        
        # Log parameters and metrics
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc_lr)
        mlflow.log_metric("f1_score", f1_lr)
        
        # Log model
        mlflow.sklearn.log_model(lr_model, "model")
        
        print(f"Logistic Regression - Accuracy: {acc_lr:.4f}, F1 Score: {f1_lr:.4f}")
        models_results['logistic_regression'] = {'accuracy': acc_lr, 'f1_score': f1_lr, 'model': lr_model}
    
    # Train and evaluate SVM
    with mlflow.start_run(run_name="SVM"):
        # Train model
        svm_model = LinearSVC(random_state=42, max_iter=2000)
        svm_model.fit(X_train, y_train)
        
        # Predict
        y_pred_svm = svm_model.predict(X_test)
        
        # Calculate metrics
        acc_svm = accuracy_score(y_test, y_pred_svm)
        f1_svm = f1_score(y_test, y_pred_svm, average='macro')
        
        # Log parameters and metrics
        mlflow.log_param("max_iter", 2000)
        mlflow.log_metric("accuracy", acc_svm)
        mlflow.log_metric("f1_score", f1_svm)
        
        # Log model
        mlflow.sklearn.log_model(svm_model, "model")
        
        print(f"SVM - Accuracy: {acc_svm:.4f}, F1 Score: {f1_svm:.4f}")
        models_results['svm'] = {'accuracy': acc_svm, 'f1_score': f1_svm, 'model': svm_model}
    
    # Determine best model based on F1 score
    best_model_name = max(models_results.items(), key=lambda x: x[1]['f1_score'])[0]
    
    # Save the name of the best model for the next task
    kwargs['ti'].xcom_push(key='best_model_name', value=best_model_name)
    kwargs['ti'].xcom_push(key='models_results', value=str(models_results))
    
    print(f"Best model: {best_model_name} with F1 score: {models_results[best_model_name]['f1_score']:.4f}")
    return f"Models trained successfully. Best model: {best_model_name}"

# Function to create recommendation engine
def create_recommendation_engine(**kwargs):
    """Create a recommendation engine based on wine attributes and subjectivity."""
    ti = kwargs['ti']
    df_json = ti.xcom_pull(key='preprocessed_df', task_ids='calculate_subjectivity')
    df = pd.read_json(df_json, orient='split')
    
    # Define the list of sensory attributes from the dataset
    attributes = [
        'frutas_rojas', 'frutas_negras', 'frutas_cítricas', 'frutas_tropicales',
        'frutas_hueso', 'frutas_secas', 'especias', 'notas_herbales_frescas',
        'notas_herbales_secas', 'notas_florales', 'notas_terrosas', 'madera_y_otros',
        'dulzor', 'acidez', 'cuerpo', 'taninos', 'alcohol', 'finalizacion', 'umami_y_otros'
    ]
    
    # Ensure all attributes exist in the DataFrame (fill with 0 if missing)
    for attr in attributes:
        if attr not in df.columns:
            df[attr] = 0.0
    
    # Define parameters for recommendation formula
    alpha = 1.0  # Weight for polarity
    beta = 0.5   # Penalty for subjectivity
    
    # Create a sample user profile for demonstration
    user_profile = {
        'frutas_rojas': 2,
        'frutas_negras': 0,
        'frutas_cítricas': -1,
        'frutas_tropicales': 1,
        'frutas_hueso': 1,
        'frutas_secas': 0,
        'especias': 2,
        'notas_herbales_frescas': 0,
        'notas_herbales_secas': -1,
        'notas_florales': 1,
        'notas_terrosas': 0,
        'madera_y_otros': 1,
        'dulzor': -1,
        'acidez': 0,
        'cuerpo': 2,
        'taninos': 1,
        'alcohol': 0,
        'finalizacion': 1,
        'umami_y_otros': 0
    }
    
    # Convert user profile to DataFrame
    user_df = pd.DataFrame([user_profile])
    
    # Calculate cosine similarity between user profile and all wines
    similarity = cosine_similarity(user_df[attributes], df[attributes])[0]
    df['similarity'] = similarity
    
    # Calculate the final score incorporating polarity and subjectivity
    df['final_score'] = (df['similarity'] + df['polarity'] * alpha) * (1 - beta * df['subjectivity'])
    
    # Filter wines with negative polarity (threshold from notebook)
    threshold = -0.2
    filtered = df[df['polarity'] > threshold]
    
    # Get top 8 recommendations
    recommendations = filtered.sort_values(by='final_score', ascending=False).head(8)
    
    # Log the recommendations to MLflow
    with mlflow.start_run(run_name="RecommendationEngine"):
        # Log user profile as parameters
        for attr, value in user_profile.items():
            mlflow.log_param(f"user_{attr}", value)
        
        # Log average similarity and score as metrics
        mlflow.log_metric("avg_similarity", recommendations['similarity'].mean())
        mlflow.log_metric("avg_final_score", recommendations['final_score'].mean())
        
        # Log the recommendations as a CSV artifact
        recommendations.to_csv('/tmp/recommendations.csv', index=False)
        mlflow.log_artifact('/tmp/recommendations.csv')
    
    # Save wine IDs of recommendations
    recommendation_ids = recommendations['wine_id'].tolist()
    kwargs['ti'].xcom_push(key='recommendation_ids', value=recommendation_ids)
    
    print(f"Created recommendations for sample user: {recommendation_ids}")
    return "Recommendation engine created successfully"

# Define the tasks
task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

task_calculate_subjectivity = PythonOperator(
    task_id='calculate_subjectivity',
    python_callable=calculate_subjectivity,
    provide_context=True,
    dag=dag,
)

task_prepare_features = PythonOperator(
    task_id='prepare_features',
    python_callable=prepare_features,
    provide_context=True,
    dag=dag,
)

task_train_models = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)

task_create_recommendation = PythonOperator(
    task_id='create_recommendation_engine',
    python_callable=create_recommendation_engine,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
task_load_data >> task_calculate_subjectivity >> task_prepare_features >> task_train_models >> task_create_recommendation