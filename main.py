from data_handler import DataHandler
from feature_engineering import FeatureEngineering
from model import FraudModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from logger import get_logger
import pickle
import pandas as pd
# General logger
logger = get_logger(__name__)

# Model-specific logger
model_logger = get_logger('model_logger', log_file='logs/model_training.log')

def main():
    try:
        # Folder containing .pkl files
        folder_path = 'Dataset'

        # Step 1: Load Data
        logger.info("Starting data loading process.")
        handler = DataHandler(folder_path)
        df = handler.load_data()

        if df is None or df.empty:
            logger.error("No data loaded. Exiting the process.")
            return

        # Step 2: Feature Engineering
        logger.info("Starting feature engineering process.")
        df = FeatureEngineering.preprocess(df)

        # Step 3: Feature and Target Selection
        features = ['TX_AMOUNT', 'DAY', 'HOUR', 'TERMINAL_FRAUD_RATIO', 'CUSTOMER_FRAUD_RATIO']
        target = 'TX_FRAUD'
        X, y = df[features], df[target]

        # Step 4: Train-Test Split
        logger.info("Splitting data into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Step 5: Apply SMOTE to balance training data
        logger.info("Applying SMOTE to balance the training dataset.")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        logger.info(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
        logger.info(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        # Step 6: Train Model
        logger.info("Starting model training.")
        model = FraudModel()
        model.train(X_train_resampled, y_train_resampled)

        # Log model training information
        model_logger.info("Model training completed successfully.")
        model_logger.info("Training dataset size: %d", len(y_train_resampled))
        model_logger.info("Testing dataset size: %d", len(y_test))

        # Save the trained model to a file
        model_path = 'fraud_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model.model, f)
            logger.info(f"Trained model saved to '{model_path}'.")
            model_logger.info(f"Trained model saved to '{model_path}'.")

        # Step 7: Evaluate Model
        logger.info("Starting model evaluation.")
        evaluation_metrics = model.evaluate(X_test, y_test)

        # Log detailed evaluation metrics
        model_logger.info("Model evaluation completed successfully.")
        model_logger.info("Evaluation Metrics:")
        model_logger.info("Accuracy Score: %.4f", evaluation_metrics["Accuracy Score"])
        model_logger.info("Confusion Matrix: %s", evaluation_metrics["Confusion Matrix"])

        # Log classification report in detail
        classification_report = evaluation_metrics["Classification Report"]
        for label, metrics in classification_report.items():
            if isinstance(metrics, dict):  # Skip the overall accuracy key
                model_logger.info(f"Label: {label}")
                for metric, value in metrics.items():
                    model_logger.info(f"  {metric}: {value:.4f}")

        logger.info("Process completed successfully.")
    except Exception as e:
        logger.critical(f"Unexpected error in main process: {e}")

if __name__ == "__main__":
    main()
