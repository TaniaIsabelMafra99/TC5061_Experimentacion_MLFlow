import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento CLI con registro en MLflow")
    parser.add_argument("--data_path", type=str, default="data/raw/yellow_tripdata_2026-08.parquet", help="Ruta al archivo Parquet")
    parser.add_argument("--n_estimators", type=int, default=100, help="Número de árboles en RandomForest")
    parser.add_argument("--max_depth", type=int, default=10, help="Profundidad máxima de los árboles")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--experiment_name", type=str, default="TC5061_Taxi_Fare_Prediction", help="Nombre del experimento MLflow")
    return parser.parse_args()

def load_and_preprocess_data(data_path):
    print(f"Cargando datos desde: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Limpieza básica
    df = df.dropna(subset=['trip_distance', 'fare_amount'])
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]
    
    # Selección de variables
    features = ['trip_distance', 'passenger_count']
    for col in ['PULocationID', 'DOLocationID']:
        if col in df.columns:
            features.append(col)
        
    X = df[features].fillna(0)
    y = df['fare_amount']
    return X, y

def main():
    args = parse_args()
    
    # Configurar MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        print("Cargando y procesando dataset...")
        X, y = load_and_preprocess_data(args.data_path)
        
        # Split con semilla fija para reproducibilidad
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state
        )
        
        # Registrar hiperparámetros
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("data_path", args.data_path)
        
        # Entrenamiento
        print(f"Entrenando modelo (n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluación
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Métricas obtenidas -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        
        # Registrar métricas en MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Crear y registrar gráfica de predicciones
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test[:1000], y=predictions[:1000], alpha=0.5, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Predicciones")
        ax.set_title("Reales vs Predicciones (Muestra)")
        
        plot_path = "predictions_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        mlflow.log_artifact(plot_path)
        if os.path.exists(plot_path):
            os.remove(plot_path)
            
        # Log del modelo con mlflow.sklearn
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )
        print("¡Ejecución registrada con éxito en MLflow!")

if __name__ == "__main__":
    main()