import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    binary_columns = ['FastingBS', 'HeartDisease']

    categorical_df = data[categorical_columns]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(categorical_df)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    numerical_df = data[numerical_columns]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)

    data_preprocessed = pd.concat([encoded_df, scaled_df], axis=1)

    data_preprocessed[binary_columns] = data[binary_columns]

    output_file_path = 'data/heart_preprocessed.csv'
    data_preprocessed.to_csv(output_file_path, index=False)

    X = data_preprocessed.drop('HeartDisease', axis=1)
    y = data_preprocessed['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)
    
    return X_train, X_test, y_train, y_test

