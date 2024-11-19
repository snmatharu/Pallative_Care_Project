import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

file_path = 'supply-distribution.csv'

with open(file_path, 'r', encoding='ISO-8859-1') as file:
    supply = pd.read_csv(file , na_values=['NA', 'null', ''])
supply = supply[['Instructions ', 'Health region', 'Specialty ',
       'Specialty \nsort', 'Physician-to 100,000 population ratio',
       'Number of physicians', 'Numbermale', 'Numberfemale',
       'Average age', 'Median age',
       'Statistics Canada population']]

province_map = {
    'N.L.': 'Newfoundland and Labrador',
    'P.E.I.': 'Prince Edward Island',
    'N.S.': 'Nova Scotia',
    'N.B.': 'New Brunswick',
    'Que.': 'Quebec',
    'Ont.': 'Ontario',
    'Man.': 'Manitoba',
    'Sask.': 'Saskatchewan',
    'Alta.': 'Alberta',
    'B.C.': 'British Columbia',
    'Y.T.': 'Yukon',
    'N.W.T.': 'Northwest Territories',
    'Nun.': 'Nunavut'
}

supply['Health region'] = supply['Health region'].replace(province_map)
unique_provinces = supply['Health region'].unique()

# Print unique provinces/territories
print(unique_provinces)
supply['Instructions '] = supply['Instructions '].astype(str)
supply['Health region'] = supply['Health region'].astype(str)
supply['Instructions '].fillna('', inplace=True)
supply['Health region'].fillna('', inplace=True)

specialty_to_category = {
    # Oncology and Cancer Care
    "__Medical oncology": "Oncology and Cancer Care",
    "__Hematology": "Oncology and Cancer Care",
    "__Gynecological oncology": "Oncology and Cancer Care",
    "__General surgical oncology": "Oncology and Cancer Care",
    "__Hematology/oncology \x97 Pediatrics": "Oncology and Cancer Care",
    "_Radiation oncology": "Oncology and Cancer Care",
    "_Otolaryngology \x97 Head and neck surgery": "Oncology and Cancer Care",
    # Surgical Interventions
    "_Cardiac surgery": "Surgical Interventions",
    "_General surgery": "Surgical Interventions",
    "__Pediatric surgery": "Surgical Interventions",
    "_Neurosurgery": "Surgical Interventions",
    "_Obstetrics and gynecology": "Surgical Interventions",
    "_Ophthalmology": "Cataract Surgery",  # Special case, could also be "Surgical Interventions"
    "_Orthopedic surgery": "Surgical Interventions",
    "_Otolaryngology \x97 Head and neck surgery": "Surgical Interventions",
    "_Plastic surgery": "Surgical Interventions",
    "_Urology": "Surgical Interventions",
    "_Vascular surgery": "Surgical Interventions",
    "__Colorectal surgery": "Surgical Interventions",

    # Diagnostic Imaging
    "_Diagnostic radiology": "Diagnostic Imaging",
    "_Medical genetics and genomics": "Diagnostic Imaging",  # Generally not but included for completeness
    "_Neurology": "Diagnostic Imaging",  # In the context of Electroencephalography
    "_Nuclear medicine": "Diagnostic Imaging",
    "__Pediatric radiology": "Diagnostic Imaging",
    "_Diagnostic and molecular pathology": "Diagnostic Imaging",
    "_Diagnostic and clinical pathology": "Diagnostic Imaging",
    "_Hematological pathology": "Diagnostic Imaging",
    "__Forensic pathology": "Diagnostic Imaging",

    # Emergency and Critical Care
    "_Emergency medicine": "Emergency and Critical Care",
    "__Critical care medicine": "Emergency and Critical Care",
    "__Emergency family medicine": "Emergency and Critical Care",
    "__Emergency medicine \x97 Pediatrics": "Emergency and Critical Care",
    "__Critical care medicine \x97 Pediatrics": "Emergency and Critical Care",

    # Cataract Surgery
    "_Ophthalmology": "Cataract Surgery",

    # Defaulting others to Other if not falling into the above categories
    "All physicians": "Essentials",
    "All specialists": "Essentials",
    "__Family medicine": "Essentials",
    "__General practice": "Essentials",
    "_Anesthesiology": "Essentials",
    "_Dermatology": "Other",
    "_Internal medicine": "Essentials",
    "__Cardiology": "Other",
    "__Clinical immunology and allergy": "Essentials",
    "__Endocrinology and metabolism": "Essentials",
    "__Gastroenterology": "Essentials",
    "__Geriatric medicine": "Essentials",
    "__Infectious diseases": "Essentials",
    "__Nephrology": "Other",
    "__Occupational medicine": "Other",
    "__Respirology": "Essentials",
    "__Rheumatology": "Other",
    "_Public health and preventive medicine": "Essentials",
    "_Physical medicine and rehabilitation": "Essentials",
    "_Psychiatry": "Essentials",
    
    "_Medical biochemistry": "Other",
    "_Medical microbiology": "Other",
    "_Neuropathology": "Other",

    # Additional pediatric specialties
    "_Pediatrics": "Other",
    "__Cardiology \x97 Pediatrics": "Other",
    "__Neonatal–perinatal medicine": "Other",
    "__Endocrinology and metabolism \x97 Pediatrics": "Other",
    "__Gastroenterology \x97 Pediatrics": "Other",
    "__Infectious diseases \x97 Pediatrics": "Other",
    "__Nephrology \x97 Pediatrics": "Other",
    "__Respirology \x97 Pediatrics": "Other",
    "__Rheumatology \x97 Pediatrics": "Other",
    "__Adolescent medicine \x97 Pediatrics": "Other",
    "__Child and adolescent psychiatry \x97 Pediatrics": "Other",
    "__General internal medicine": "Other",
    "__Forensic psychiatry": "Other",
    "__Maternal–fetal medicine": "Other",
    "__Neuroradiology": "Other",
    "__Developmental \x97 Pediatrics": "Other",
    "__Geriatric psychiatry": "Other",
    
    # Newly added specialties
    "__Electroencephalography": "Other",
    "__Cardiology \x97 Pediatrics": "Other",
    "__Neonatal\x96perinatal medicine": "Other",
    
    "Medical scientists": "Essentials",
    "__Palliative medicine": "Essentials",
    "__Clinical immunology and allergy \x97 Pediatrics": "Other",
    "__Endocrinology and metabolism \x97 Pediatrics": "Other",
    "__Gastroenterology \x97 Pediatrics": "Other",
    "__Hematology/oncology \x97 Pediatrics": "Other",
    "__Infectious diseases \x97 Pediatrics": "Other",
    "__Nephrology \x97 Pediatrics": "Other",
    "__Respirology \x97 Pediatrics": "Other",
    "__Rheumatology \x97 Pediatrics": "Other",
    "__Emergency medicine \x97 Pediatrics": "Other",
    "__Critical care medicine \x97 Pediatrics": "Other",
    "__Adolescent medicine \x97 Pediatrics": "Other",
    "__Child and adolescent psychiatry \x97 Pediatrics": "Other",
    "__Maternal\x96fetal medicine": "Essentials",
    "__Developmental \x97 Pediatrics": "Other"
}

# Replace the procedure names in the DataFrame
supply['Category'] = supply['Specialty '].map(specialty_to_category).fillna(supply['Specialty '])

# Print the updated DataFrame
# display(supply)
supply['SupplyKey'] = supply['Health region'] + ' ' + (supply['Instructions ']) +' ' + (supply['Category'])
# display(supply)
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

# Reset the index if necessary to flatten the DataFrame
pivot_df.reset_index(inplace=True)
pivot_df = pivot_df[['Province/territory', 'Indicator', 'Data year','50thPercentile','90thPercentile', 'Volume']]
# Display the resulting DataFrame
# display(pivot_df)
# Dictionary mapping old procedures to new category names
procedure_to_category = {
    "CT Scan": "Diagnostic Imaging",
    "MRI Scan": "Diagnostic Imaging",
    "Cataract Surgery": "Diagnostic Imaging",
    "Bladder Cancer Surgery": "Surgical Interventions",
    "Breast Cancer Surgery": "Surgical Interventions",
    "CABG": "Surgical Interventions",
    "Colorectal Cancer Surgery": "Surgical Interventions",
    "Hip Fracture Repair": "Surgical Interventions",
    "Hip Replacement": "Surgical Interventions",
    "Knee Replacement": "Surgical Interventions",
    "Lung Cancer Surgery": "Surgical Interventions",
    "Prostate Cancer Surgery": "Surgical Interventions",
    "Bladder Cancer Surgery": "Oncology and Cancer Care",
    "Breast Cancer Surgery": "Oncology and Cancer Care",
    "Colorectal Cancer Surgery": "Oncology and Cancer Care",
    "Lung Cancer Surgery": "Oncology and Cancer Care",
    "Prostate Cancer Surgery": "Oncology and Cancer Care",
    "Radiation Therapy": "Oncology and Cancer Care",
    "Hip Fracture Repair/Emergency and Inpatient": "Emergency and Critical Care"
}

# Replace the procedure names in the DataFrame
pivot_df['Category'] = pivot_df['Indicator'].map(procedure_to_category).fillna(pivot_df['Indicator'])

# Print the updated DataFrame
# print(pivot_df)
pivot_df = pivot_df[pivot_df['Data year'].str.len() == 4]
pivot_df = pivot_df.sort_values(by='Data year')

# Display the resulting DataFrame
# print(pivot_df)
pivot_df['WaitKey'] = pivot_df['Province/territory'] + ' ' + (pivot_df['Data year']) +' ' + (pivot_df['Category'])
# display(pivot_df)
l1 = pivot_df['WaitKey'].unique()
l2 = supply['SupplyKey'].unique()
unique_items = set(l1).symmetric_difference(set(l2))
print(unique_items)
merged_df = pd.merge(supply, pivot_df, left_on='SupplyKey', right_on='WaitKey', suffixes=('SupplyKey', 'WaitKey'), how='inner')

# Display the result
# print(merged_df)
# 0 represents that there was no waiting time that day 
merged_df.fillna(0, inplace=True) 
merged_df = (merged_df[['SupplyKey','Specialty ','Indicator','Number of physicians',
          'Physician-to 100,000 population ratio', 'CategorySupplyKey', 
          'Province/territory',  'Data year', '50thPercentile',
          '90thPercentile', 'Volume', 'Statistics Canada population','Numbermale', 
          'Numberfemale', 'Average age', 'Median age']])
def scale_medical_data(df):
    """
    Scale numerical columns in the medical dataset using StandardScaler.
    Handles string numbers with commas.
    """
    # Create a copy of the DataFrame
    scaled_df = df.copy()
    
    # Identify numerical columns to scale
    numerical_columns = [
        'Number of physicians',
        'Physician-to 100,000 population ratio',
        '50thPercentile',
        '90thPercentile',
        'Volume',
        'Statistics Canada population',
        'Numbermale',
        'Numberfemale',
        'Average age',
        'Median age'
    ]
    
    # Convert string numbers with commas to float
    for col in numerical_columns:
        if scaled_df[col].dtype == 'object':
            scaled_df[col] = scaled_df[col].replace({',': ''}, regex=True).astype(float)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Scale numerical columns
    scaled_data = scaler.fit_transform(scaled_df[numerical_columns])
    
    # Replace original columns with scaled values
    for i, col in enumerate(numerical_columns):
        scaled_df[col] = scaled_data[:, i]
    
    # Add feature names for scaled columns
    scaled_columns = {col: f'{col}_scaled' for col in numerical_columns}
    scaled_df = scaled_df.rename(columns=scaled_columns)
    
    # Create a dictionary to store scaling parameters
    scaling_params = {
        column: {
            'mean': scaler.mean_[i],
            'scale': scaler.scale_[i]
        }
        for i, column in enumerate(numerical_columns)
    }
    
    return scaled_df, scaling_params

# Apply scaling to the merged dataset
scaled_merged_df, scaling_parameters = scale_medical_data(merged_df)

# Create summary statistics of scaled data
summary_stats = scaled_merged_df.describe()

# Function to check for any remaining outliers
def check_outliers(df, scaled_columns, threshold=3):
    """
    Check for outliers in scaled data using z-score method.
    """
    outliers = {}
    for col in scaled_columns:
        if col.endswith('_scaled'):
            outlier_count = len(df[abs(df[col]) > threshold])
            if outlier_count > 0:
                outliers[col] = outlier_count
    return outliers

# Get scaled column names
scaled_columns = [col for col in scaled_merged_df.columns if col.endswith('_scaled')]

# Check for outliers in scaled data
outliers = check_outliers(scaled_merged_df, scaled_columns)

print("Scaling completed successfully!")
print("\nScaling Parameters:")
for col, params in scaling_parameters.items():
    print(f"\n{col}:")
    print(f"Mean: {params['mean']:.2f}")
    print(f"Scale: {params['scale']:.2f}")

print("\nOutliers detected (|z-score| > 3):")
for col, count in outliers.items():
    print(f"{col}: {count} outliers")
def encode_categorical_data(df):
    """
    Encode categorical columns using LabelEncoder while preserving original values.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with medical data
    
    Returns:
    pandas.DataFrame: DataFrame with encoded categorical columns
    dict: Dictionary with encoding mappings
    """
    # Create a copy of the DataFrame
    encoded_df = df.copy()
    
    # Identify categorical columns
    categorical_columns = [
        'SupplyKey',
        'Specialty ',
        'Indicator',
        'CategorySupplyKey',
        'Province/territory',
        'Data year'
    ]
    
    # Dictionary to store encoders and mappings
    encoding_mappings = {}
    
    # Encode each categorical column
    for column in categorical_columns:
        if column in encoded_df.columns:
            # Create a label encoder for the column
            le = LabelEncoder()
            
            # Fit and transform the column
            encoded_df[f'{column}_encoded'] = le.fit_transform(encoded_df[column].astype(str))
            
            # Store the mapping
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            encoding_mappings[column] = mapping
            
            # Store reverse mapping for later use
            encoding_mappings[f'{column}_reverse'] = dict(zip(le.transform(le.classes_), le.classes_))
    
    return encoded_df, encoding_mappings

# Function to decode the encoded values
def decode_categorical_data(encoded_df, encoding_mappings, column_name):
    """
    Decode encoded categorical values back to original values.
    
    Parameters:
    encoded_df (pandas.DataFrame): DataFrame with encoded values
    encoding_mappings (dict): Dictionary with encoding mappings
    column_name (str): Name of the column to decode
    
    Returns:
    pandas.Series: Decoded values
    """
    if f'{column_name}_encoded' in encoded_df.columns:
        reverse_mapping = encoding_mappings[f'{column_name}_reverse']
        return encoded_df[f'{column_name}_encoded'].map(reverse_mapping)
    return None

# Apply encoding to the scaled dataset
encoded_df, encoding_mappings = encode_categorical_data(scaled_merged_df)

# Print sample of encoding mappings
print("Sample of encoding mappings:")
for column, mapping in encoding_mappings.items():
    if not column.endswith('_reverse'):
        print(f"\n{column}:")
        # Print first 5 mappings for each column
        for original, encoded in list(mapping.items())[:5]:
            print(f"{original} -> {encoded}")

# Verify we can decode back to original values
print("\nVerification of encoding/decoding:")
for column in encoding_mappings.keys():
    if not column.endswith('_reverse'):
        # Take a sample record and verify encoding/decoding
        original_value = scaled_merged_df[column].iloc[0]
        encoded_value = encoded_df[f'{column}_encoded'].iloc[0]
        decoded_value = decode_categorical_data(encoded_df, encoding_mappings, column).iloc[0]
        
        print(f"\n{column}:")
        print(f"Original: {original_value}")
        print(f"Encoded: {encoded_value}")
        print(f"Decoded: {decoded_value}")

# Create summary statistics of encoded categorical columns
encoded_columns = [col for col in encoded_df.columns if col.endswith('_encoded')]
categorical_summary = encoded_df[encoded_columns].describe()
print("\nSummary statistics of encoded categorical columns:")
print(categorical_summary)
# Get overall null count
total_nulls = merged_df.isnull().sum()

# Get percentage of nulls
null_percentage = (merged_df.isnull().sum() / len(merged_df)) * 100

# Combine into a summary DataFrame
null_summary = pd.DataFrame({
    'Null Count': total_nulls,
    'Null Percentage': null_percentage
})

# Sort by null count descending
null_summary = null_summary.sort_values('Null Count', ascending=False)

# Only show columns with null values
null_summary = null_summary[null_summary['Null Count'] > 0]

print("Null Value Summary:")
print(null_summary)

# Check for any rows that are completely null
complete_null_rows = merged_df[merged_df.isnull().all(axis=1)]
print(f"\nNumber of completely null rows: {len(complete_null_rows)}")

# Check for rows with more than 50% null values
threshold = 0.5
rows_with_many_nulls = merged_df[merged_df.isnull().mean(axis=1) > threshold]
print(f"Number of rows with more than {threshold*100}% null values: {len(rows_with_many_nulls)}")

# Get sample rows with null values
print("\nSample rows with null values:")
sample_nulls = merged_df[merged_df.isnull().any(axis=1)].head()
print(sample_nulls)

# Check for patterns in null values
# For example, are nulls more common in certain provinces or categories?
if null_summary.shape[0] > 0:  # If there are any null values
    print("\nNull value patterns by province:")
    province_nulls = merged_df.groupby('Province/territory')[merged_df.columns].isnull().mean() * 100
    print(province_nulls)
    
    print("\nNull value patterns by category:")
    category_nulls = merged_df.groupby('CategorySupplyKey')[merged_df.columns].isnull().mean() * 100
    print(category_nulls)
# First, let's check the actual column names
print("Available columns in scaled_merged_df:")
print(scaled_merged_df.columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_correlation_analysis(df):
    """
    Create comprehensive correlation analysis for numerical variables.
    """
    # Get all columns that end with '_scaled'
    numerical_cols = [col for col in df.columns if col.endswith('_scaled')]
    
    # Get all columns that end with '_encoded'
    categorical_cols = [col for col in df.columns if col.endswith('_encoded')]
    
    # Combine numerical and encoded categorical columns
    analysis_cols = numerical_cols + categorical_cols
    
    # Calculate correlation matrix
    correlation_matrix = df[analysis_cols].corr()
    
    # Create figure and axis
    plt.figure(figsize=(15, 12))
    
    # Create heatmap using plt.imshow
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add labels
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    
    # Add title
    plt.title('Correlation Matrix of Medical Data\n(Numerical and Categorical Variables)')
    
    # Add correlation values as text
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.index)):
            text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return correlation_matrix

# Create correlation analysis
correlation_matrix = create_correlation_analysis(scaled_merged_df)

# Find and print strongest correlations
threshold = 0.5  # Define threshold here
print(f"\nStrongest Correlations (|correlation| > {threshold}):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > threshold:
            var1 = correlation_matrix.index[i]
            var2 = correlation_matrix.columns[j]
            print(f"{var1} vs {var2}: {corr:.3f}")

plt.show()

# Keep only numerical columns that end with '_scaled'
numerical_columns = [col for col in scaled_merged_df.columns if col.endswith('_scaled')]
X = scaled_merged_df[numerical_columns].drop('Number of physicians_scaled', axis=1)
y = scaled_merged_df['Number of physicians_scaled']

# Feature selection using correlation with target
correlations = abs(X.corrwith(y)).sort_values(ascending=False)
print("Feature Correlations with Target:")
print(correlations)

# Select top 5 features based on correlation
top_features = correlations.head(5).index.tolist()
X_selected = X[top_features]

print("\nSelected Features:")
print(top_features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Try multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Compare models
for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross validation
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
    
    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Cross-validation R² scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    if name == 'Random Forest':
        importance = pd.DataFrame({
            'Feature': X_selected.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nRandom Forest Feature Importance:")
        print(importance)
    elif name == 'Linear Regression':
        importance = pd.DataFrame({
            'Feature': X_selected.columns,
            'Coefficient': abs(model.coef_)
        }).sort_values('Coefficient', ascending=False)
        print("\nLinear Regression Coefficients:")
        print(importance)

# Calculate and print residuals statistics for best model
best_model = models['Random Forest']  # Usually performs better
y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

print("\nResiduals Statistics:")
print(f"Mean residual: {np.mean(residuals):.4f}")
print(f"Std residual: {np.std(residuals):.4f}")
print(f"Min residual: {np.min(residuals):.4f}")
print(f"Max residual: {np.max(residuals):.4f}")
# List of columns to remove
columns_to_remove = [
    'Numbermale_scaled',
    'Numberfemale_scaled',
    'Statistics Canada population_scaled'
]

# Keep only numerical columns that end with '_scaled'
numerical_columns = [col for col in scaled_merged_df.columns if col.endswith('_scaled')]
numerical_df = scaled_merged_df[numerical_columns]

# Remove highly correlated features
model_df = numerical_df.drop(columns=columns_to_remove)

# Verify remaining correlations
remaining_correlations = model_df.corr()['Number of physicians_scaled'].sort_values(ascending=False)
print("\nRemaining correlations with Number of physicians_scaled:")
print(remaining_correlations)

# Create and evaluate model
X = model_df.drop('Number of physicians_scaled', axis=1)
y = model_df['Number of physicians_scaled']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Compare models
for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Cross-validation R² scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    if name == 'Random Forest':
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nRandom Forest Feature Importance:")
        print(importance)
    elif name == 'Linear Regression':
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': abs(model.coef_)
        }).sort_values('Coefficient', ascending=False)
        print("\nLinear Regression Coefficients:")
        print(importance)

# Use the best performing model (Random Forest)
best_model = models['Random Forest']
y_pred_best = best_model.predict(X_test)

# Print prediction examples
print("\nSample Predictions vs Actual Values:")
sample_comparison = pd.DataFrame({
    'Actual': y_test.iloc[:5],
    'Predicted': y_pred_best[:5],
    'Difference': y_test.iloc[:5] - y_pred_best[:5]
})
print(sample_comparison)

# Calculate residuals statistics
residuals = y_test - y_pred_best
print("\nResiduals Statistics:")
print(f"Mean residual: {np.mean(residuals):.4f}")
print(f"Std residual: {np.std(residuals):.4f}")
print(f"Min residual: {np.min(residuals):.4f}")
print(f"Max residual: {np.max(residuals):.4f}")

# Print final features used
print("\nFinal Features Used:")
for col in X.columns:
    print(f"- {col}")
    
# Assuming scaled_merged_df is your input DataFrame
def analyze_healthcare_staffing(scaled_merged_df):
    # List of columns to remove
    columns_to_remove = [
        'Numbermale_scaled',
        'Numberfemale_scaled',
        'Statistics Canada population_scaled'
    ]
    
    # Keep only numerical columns that end with '_scaled'
    numerical_columns = [col for col in scaled_merged_df.columns if col.endswith('_scaled')]
    numerical_df = scaled_merged_df[numerical_columns]
    
    # Remove highly correlated features
    model_df = numerical_df.drop(columns=columns_to_remove)
    
    # Verify remaining correlations
    remaining_correlations = model_df.corr()['Number of physicians_scaled'].sort_values(ascending=False)
    print("\nRemaining correlations with Number of physicians_scaled:")
    print(remaining_correlations)
    
    # Create features and target
    X = model_df.drop('Number of physicians_scaled', axis=1)
    y = model_df['Number of physicians_scaled']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Plot setup for learning curves
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    # Compare models
    for idx, (name, model) in enumerate(models.items()):
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Store results
        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Cross-validation R² scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Plot learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train_scaled, y_train, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, label='Training score')
        axes[idx].plot(train_sizes, test_mean, label='Cross-validation score')
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        axes[idx].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        axes[idx].set_title(f'Learning Curve - {name}')
        axes[idx].set_xlabel('Training Examples')
        axes[idx].set_ylabel('Score')
        axes[idx].legend(loc='best')
        axes[idx].grid(True)
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            print(f"\n{name} Feature Importance:")
            print(importance)
        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': abs(model.coef_)
            }).sort_values('Coefficient', ascending=False)
            print(f"\n{name} Coefficients:")
            print(importance)
    
    plt.tight_layout()
    plt.show()
    
    # Compare model performances
    performance_comparison = pd.DataFrame({
        'RMSE': [results[model]['rmse'] for model in results],
        'R²': [results[model]['r2'] for model in results],
        'MAE': [results[model]['mae'] for model in results],
        'CV Mean R²': [results[model]['cv_mean'] for model in results],
        'CV Std R²': [results[model]['cv_std'] for model in results]
    }, index=results.keys())
    
    # Plot model comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RMSE Comparison
    performance_comparison['RMSE'].plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('RMSE by Model')
    axes[0,0].set_ylabel('RMSE')
    
    # R² Comparison
    performance_comparison['R²'].plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('R² Score by Model')
    axes[0,1].set_ylabel('R²')
    
    # Prediction vs Actual Plot for best model
    best_model_name = performance_comparison['R²'].idxmax()
    best_predictions = results[best_model_name]['predictions']
    
    axes[1,0].scatter(y_test, best_predictions, alpha=0.5)
    axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1,0].set_title(f'Actual vs Predicted ({best_model_name})')
    axes[1,0].set_xlabel('Actual')
    axes[1,0].set_ylabel('Predicted')
    
    # Residuals Plot
    residuals = y_test - best_predictions
    axes[1,1].scatter(best_predictions, residuals, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_title('Residuals Plot')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\nModel Performance Comparison:")
    print(performance_comparison)
    
    # Calculate residuals statistics for best model
    print(f"\nBest Model ({best_model_name}) Residuals Statistics:")
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Std residual: {np.std(residuals):.4f}")
    print(f"Min residual: {np.min(residuals):.4f}")
    print(f"Max residual: {np.max(residuals):.4f}")
    
    # Print final features used
    print("\nFinal Features Used:")
    for col in X.columns:
        print(f"- {col}")
    
    return results, performance_comparison

# Usage example (assuming you have scaled_merged_df):
results, performance_comparison = analyze_healthcare_staffing(scaled_merged_df)