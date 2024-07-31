import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load the model
model = pickle.load(open('model.pkl', 'rb'))

# read the input 
df_test = pd.read_csv('hr_attrition_ulabeled_test.csv')
# drop Atttition label
df_test = df_test.drop(columns='Attrition', axis=1)

# processing dataset
# encode categorical features
to_encode = df_test.select_dtypes(include=['object']).columns

# initialize LabelEncoder
label_encoder = LabelEncoder()

# encode each categorical column
for col in to_encode:
    df_test[col] = label_encoder.fit_transform(df_test[col])

# slice one row from the processed data test
test = df_test[52:53]

def encode_label(prediction):
    """
    Encode the prediction result into human-readable labels.

    Parameters:
    prediction (int): The prediction result (0 or 1).

    Returns:
    str: The human-readable label corresponding to the prediction.
    """
    if prediction == 0:
        return "Didnt leave"
    elif prediction == 1:
        return "Left"
    else:
        return "Invalid prediction"

# prediction
prediction = loaded_model.predict(test)
prediction = prediction[0]
result = encode_label(prediction)

print('Predicted Status:', result)