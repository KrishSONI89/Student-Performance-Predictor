import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 5000

data = {
    'Hours_Studied': np.random.randint(1, 10, n_samples),
    'Previous_Scores': np.random.randint(40, 100, n_samples),
    'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_samples),
    'Sleep_Hours': np.random.randint(4, 10, n_samples),
    'Sample_Question_Papers_Practiced': np.random.randint(0, 10, n_samples)
}
df = pd.DataFrame(data)

# Generate target variable: Performance Index
df['Performance_Index'] = (
    (df['Hours_Studied'] * 2.5) + 
    (df['Previous_Scores'] * 0.6) + 
    (df['Sleep_Hours'] * 1.5) + 
    (df['Sample_Question_Papers_Practiced'] * 1.0) +
    np.where(df['Extracurricular_Activities'] == 'Yes', 3, 0) + 
    np.random.normal(0, 2, n_samples)
).round(2)

df['Performance_Index'] = np.clip(df['Performance_Index'], 10, 100)
df.to_csv('student_performance.csv', index=False)
print("Dataset generated successfully at 'student_performance.csv'")