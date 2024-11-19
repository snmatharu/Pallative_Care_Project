import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from mappings import procedure_to_category, specialty_to_category, province_map
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class WorkerPredictionModel:
    def __init__(self):
        
        # Merge and preprocess data
        self.merged_df = self._load_data()
        
        # Scaling and encoding
        self.scaled_merged_df, self.scaling_parameters = self.scale_medical_data(self.merged_df)
        self.encoded_df, self.encoding_mappings = self.encode_categorical_data(self.scaled_merged_df)
        
        # Feature selection
        self.X_selected, self.y = self._feature_selection()
        
        # Model initialization
        self.model = RandomForestRegressor()

    def _load_data(self):
        file_path = 'supply-distribution.csv'

        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            supply = pd.read_csv(file , na_values=['NA', 'null', ''])
        supply = supply[['Instructions ', 'Health region', 'Specialty ',
            'Specialty \nsort', 'Physician-to 100,000 population ratio',
            'Number of physicians', 'Numbermale', 'Numberfemale',
            'Average age', 'Median age',
            'Statistics Canada population']]

        supply['Health region'] = supply['Health region'].replace(province_map)
        supply['Instructions '] = supply['Instructions '].astype(str)
        supply['Health region'] = supply['Health region'].astype(str)
        supply['Instructions '].fillna('', inplace=True)
        supply['Health region'].fillna('', inplace=True)

        # Replace the procedure names in the DataFrame
        supply['Category'] = supply['Specialty '].map(specialty_to_category).fillna(supply['Specialty '])
        supply['SupplyKey'] = supply['Health region'] + ' ' + (supply['Instructions ']) +' ' + (supply['Category'])
        supply[supply["Category"]=="Essentials"].groupby('SupplyKey')[['Number of physicians']].sum()
        file_path = 'wait times edit.csv'

        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            waitTime = pd.read_csv(file, na_values=['NA', 'null', ''])
        pivot_df = waitTime.pivot_table(
            index=['Reporting level', 'Province/territory', 'Indicator', 'Data year'],
            columns='Metric',
            values='Indicator result',
            aggfunc='first'
        )
        pivot_df.reset_index(inplace=True)
        pivot_df = pivot_df[['Province/territory', 'Indicator', 'Data year','50thPercentile','90thPercentile', 'Volume']]
    
        # Replace the procedure names in the DataFrame
        pivot_df['Category'] = pivot_df['Indicator'].map(procedure_to_category).fillna(pivot_df['Indicator'])
        pivot_df = pivot_df[pivot_df['Data year'].str.len() == 4]
        pivot_df = pivot_df.sort_values(by='Data year')
        pivot_df['WaitKey'] = pivot_df['Province/territory'] + ' ' + (pivot_df['Data year']) +' ' + (pivot_df['Category'])
        merged_df = pd.merge(supply, pivot_df, left_on='SupplyKey', right_on='WaitKey', suffixes=('SupplyKey', 'WaitKey'), how='inner')
        print(merged_df.columns)
        return merged_df

    

    def scale_medical_data(self, df):
        # Scale numerical columns
        numerical_columns = [
            'Number of physicians', 'Physician-to 100,000 population ratio', '50thPercentile', '90thPercentile', 'Volume', 'Statistics Canada population', 'Numbermale', 'Numberfemale', 'Average age', 'Median age'
        ]
        
        for col in numerical_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace({',': ''}, regex=True).astype(float)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_columns])
        
        for i, col in enumerate(numerical_columns):
            df[f'{col}_scaled'] = scaled_data[:, i]
        
        return df, {col: {'mean': scaler.mean_[i], 'scale': scaler.scale_[i]} for i, col in enumerate(numerical_columns)}

    def encode_categorical_data(self, df):
        # Encode categorical columns
        categorical_columns = ['SupplyKey', 'Specialty ', 'Indicator', 'CategorySupplyKey', 'Province/territory', 'Data year']
        encoding_mappings = {}
        
        for column in categorical_columns:
            le = LabelEncoder()
            df[f'{column}_encoded'] = le.fit_transform(df[column].astype(str))
            encoding_mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
            encoding_mappings[f'{column}_reverse'] = dict(zip(le.transform(le.classes_), le.classes_))
        
        return df, encoding_mappings

    def _feature_selection(self):
        # Keep only scaled numerical columns
        numerical_columns = [col for col in self.encoded_df.columns if col.endswith('_scaled') or col.endswith('_encoded')]

        print(numerical_columns)
        X = self.encoded_df[numerical_columns].drop('Number of physicians_scaled', axis=1)
        y = self.encoded_df['Number of physicians_scaled']
        
        # Feature selection
        correlations = abs(X.corrwith(y)).sort_values(ascending=False)
        print(correlations)
        top_features = correlations.head(5).index.tolist()+ ['Province/territory_encoded', 'CategorySupplyKey_encoded']
        categories = {col: self.encoded_df[col].unique().tolist() for col in ['Province/territory_encoded','Province/territory','CategorySupplyKey_encoded','CategorySupplyKey']}
        print(categories)
        X_selected = X[top_features]
        return X_selected, y


    def train_and_save_model(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_selected, self.y, test_size=0.2, random_state=42)
        print(X_train.columns)
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Predictions on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate and print evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        print("Model Evaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save model, scaler, and encoder information
        joblib.dump({
            'model': self.model,
            'scaler': self.scaling_parameters,
            'encoder': self.encoding_mappings
        }, 'worker_prediction_model.pkl')
        return X_test
    


# Run training and save model
if __name__ == "__main__":
    worker_model = WorkerPredictionModel()
    Xtest = worker_model.train_and_save_model()
    print("Model and preprocessing saved successfully!")
    
