class FeatureEngineering:
    @staticmethod
    def add_terminal_fraud_ratio(df):
        """
        Adds terminal fraud ratio as a feature.
        """
        terminal_fraud_ratio = df.groupby('TERMINAL_ID')['TX_FRAUD'].mean()
        df['TERMINAL_FRAUD_RATIO'] = df['TERMINAL_ID'].map(terminal_fraud_ratio)
        return df

    @staticmethod
    def add_customer_fraud_ratio(df):
        """
        Adds customer fraud ratio as a feature.
        """
        customer_fraud_ratio = df.groupby('CUSTOMER_ID')['TX_FRAUD'].mean()
        df['CUSTOMER_FRAUD_RATIO'] = df['CUSTOMER_ID'].map(customer_fraud_ratio)
        return df

    @staticmethod
    def extract_datetime_features(df):
        """
        Extracts day and hour from the datetime column.
        """
        df['DAY'] = df['TX_DATETIME'].dt.day
        df['HOUR'] = df['TX_DATETIME'].dt.hour
        return df

    @staticmethod
    def preprocess(df):
        """
        Preprocess the dataset by adding new features.
        """
        df = FeatureEngineering.extract_datetime_features(df)
        df = FeatureEngineering.add_terminal_fraud_ratio(df)
        df = FeatureEngineering.add_customer_fraud_ratio(df)
        df.fillna(0, inplace=True)  # Replace NaN values with 0
        print("Feature engineering completed.")
        return df
