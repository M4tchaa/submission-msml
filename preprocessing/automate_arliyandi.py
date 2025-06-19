import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(input_path='PS4_GamesSales_raw.csv',
                      output_path='./game_preprocessing',
                      encoding='latin1',
                      global_sales_threshold=1.0,
                      test_size=0.2,
                      random_state=42):
    
    # Load
    df = pd.read_csv(input_path, encoding=encoding)
    
    # Impute missing values
    df['Year'].fillna(df['Year'].median(), inplace=True)
    df['Publisher'].fillna('Unknown', inplace=True)

    # Target klasifikasi
    df['Sales_Class'] = df['Global'].apply(lambda x: 1 if x >= global_sales_threshold else 0)

    # Label Encoding
    le_genre = LabelEncoder()
    le_publisher = LabelEncoder()
    
    df['Genre'] = le_genre.fit_transform(df['Genre'])
    df['Publisher'] = le_publisher.fit_transform(df['Publisher'])

    #Split data
    X = df[['Year', 'Genre', 'Publisher', 'North America', 'Europe', 'Japan', 'Rest of World']]
    y = df['Sales_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Simpan hasil
    os.makedirs(output_path, exist_ok=True)

    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)

    #Simpan encoder
    joblib.dump(le_genre, os.path.join(output_path, 'le_genre.pkl'))
    joblib.dump(le_publisher, os.path.join(output_path, 'le_publisher.pkl'))

    print("✅ Automate preprocessing Done!. Output di:", output_path)


if __name__ == "__main__":
    run_preprocessing()
