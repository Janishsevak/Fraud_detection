# Fraud_detection
This project developed by using machine learning

Fraud Detection Project

Title: Fraud Detection Using Machine Learning
Subtitle: A Project Based on Random Forest Classifier
Your Name: Jainishkumar Sevak
Date: 15-01-2025

1. Project Objective
•	Objective:
To build a robust machine learning model that detects fraudulent activities using historical data.
•	Key Goals:
o	Train a classification model using labeled data.
o	Evaluate the model's performance using various metrics

2. Dataset Description
•	Data Sources:
o	Input features (X): []
o	Target labels (y): Fraudulent (1) vs. Non-Fraudulent (0).
•	Size of Dataset : [Training=1403324, Testing = 350831]
•	Preprocessing Steps (if any):
o	Handling missing data.
o	Normalization or scaling (if applicable).

3. Methodology
•	Model Selection:
o	Random Forest Classifier with 100 estimators, random state = 42.
•	Steps Involved:
1.	Data Splitting: Train-Test Split.
2.	Model Training on Training Data.
3.	Model Evaluation on Testing Data.

4. Implementation Details
•	Programming Language & Libraries Used:
o	Python: sklearn, classification_report, confusion_matrix, accuracy_score.
•	Training:

•	# Step 6: Train Model
•	        logger.info("Starting model training.")
•	        model = FraudModel()
•	        model.train(X_train_resampled, y_train_resampled)
•	
•	        # Log model training information
•	        model_logger.info("Model training completed successfully.")
•	        model_logger.info("Training dataset size: %d", len(y_train_resampled))
•	        model_logger.info("Testing dataset size: %d", len(y_test))
•	
•	        # Save the trained model to a file
•	        model_path = 'fraud_model.pkl'
•	        with open(model_path, 'wb') as f:
•	            pickle.dump(model.model, f)
•	            logger.info(f"Trained model saved to '{model_path}'.")
•	            model_logger.info(f"Trained model saved to '{model_path}'.")

•	Evaluation:


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

5. Evaluation Metrics
•  Accuracy: 97.47%
•  Non-Fraudulent Class (Label 0):
Precision: 99.81%, Recall: 97.63%, F1-Score: 98.71%
•  Fraudulent Class (Label 1):
Precision: 21.73%, Recall: 77.83%, F1-Score: 33.97%
•  Macro Average: Precision: 60.77%, Recall: 87.73%, F1-Score: 66.34%
Evaluation Metrics:
2025-01-15 10:20:01 - INFO - Accuracy Score: 0.9747
2025-01-15 10:20:01 - INFO - Confusion Matrix: [[339664, 8231], [651, 2285]]
2025-01-15 10:20:01 - INFO - Label: 0
2025-01-15 10:20:01 - INFO -   precision: 0.9981
2025-01-15 10:20:01 - INFO -   recall: 0.9763
2025-01-15 10:20:01 - INFO -   f1-score: 0.9871
2025-01-15 10:20:01 - INFO -   support: 347895.0000
2025-01-15 10:20:01 - INFO - Label: 1
2025-01-15 10:20:01 - INFO -   precision: 0.2173
2025-01-15 10:20:01 - INFO -   recall: 0.7783
2025-01-15 10:20:01 - INFO -   f1-score: 0.3397
2025-01-15 10:20:01 - INFO -   support: 2936.0000
2025-01-15 10:20:01 - INFO - Label: macro avg
2025-01-15 10:20:01 - INFO -   precision: 0.6077
2025-01-15 10:20:01 - INFO -   recall: 0.8773
2025-01-15 10:20:01 - INFO -   f1-score: 0.6634
2025-01-15 10:20:01 - INFO -   support: 350831.0000
2025-01-15 10:20:01 - INFO - Label: weighted avg
2025-01-15 10:20:01 - INFO -   precision: 0.9916
2025-01-15 10:20:01 - INFO -   recall: 0.9747
2025-01-15 10:20:01 - INFO -   f1-score: 0.9817
2025-01-15 10:20:01 - INFO -   support: 350831.0000



6. Results
•	Key Observations:
o	Accuracy achieved: [0.9747].
o	The confusion matrix reveals that the model correctly identified 339,664 non-fraudulent cases and 2,285 fraudulent cases. It also flagged 651 false negatives and 8,231 false positives. The model demonstrates strong recall for detecting fraudulent transactions (77.83%)
o	Strengths: High recall for fraudulent transactions.


7. Challenges & Solutions
•	Challenges Faced:
o	Imbalanced dataset (fraud cases being rare).
o	Selecting optimal hyperparameters.
•	Solutions Implemented:
o	Adjusted random state and estimators for stability.
o	Utilized classification report for detailed insights.
o	Implemented SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance, increasing the number of samples for the minority class (fraudulent transactions)
8. Future Work
•	Possible Improvements:
o	Testing other classifiers (e.g., Gradient Boosting, Neural Networks).
o	Applying feature engineering to improve performance.
o	Addressing class imbalance using techniques like SMOTE.
o	Incorporating advanced ensemble methods such as XGBoost or LightGBM may improve performance. Additionally, experimenting with deep learning models like Neural Networks could yield better fraud detection results. Enhanced feature engineering, such as time-based aggregations and anomaly detection techniques, might further improve predictive accuracy





9. Conclusion
•	Summary:
The Random Forest model demonstrated strong overall performance, achieving 97.47% accuracy. It effectively identified non-fraudulent transactions with high precision (99.81%) and an F1-score of 98.71%. While the model showed limitations in precision for fraudulent cases (21.73%), it achieved a recall of 77.83%, ensuring most fraudulent cases were detected. The macro-average F1-score of 66.34% highlights room for improvement in handling imbalanced datasets.

.


