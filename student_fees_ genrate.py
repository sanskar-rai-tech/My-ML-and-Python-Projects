import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = [
    {"name": "Sanskar", "class_12_percent": 88.5, "jee_percentile": 92.3, "fees": 35000},
    {"name": "Aman Sharma", "class_12_percent": 91.2, "jee_percentile": 95.8, "fees": 25000},
    {"name": "Riya Verma", "class_12_percent": 76.4, "jee_percentile": 82.1, "fees": 90000},
    {"name": "Aditya Patel", "class_12_percent": 84.0, "jee_percentile": 88.7, "fees": 50000},
    {"name": "Priya Singh", "class_12_percent": 93.6, "jee_percentile": 97.5, "fees": 25000},
    {"name": "Rohit Kumar", "class_12_percent": 69.8, "jee_percentile": 75.4, "fees": 120000},
    {"name": "Anjali Gupta", "class_12_percent": 89.3, "jee_percentile": 90.1, "fees": 50000},
    {"name": "Vivek Yadav", "class_12_percent": 81.5, "jee_percentile": 86.9, "fees": 70000},
    {"name": "Sneha Jain", "class_12_percent": 94.2, "jee_percentile": 98.1, "fees": 15000},
    {"name": "Harsh Dubey", "class_12_percent": 72.5, "jee_percentile": 78.3, "fees": 90000},
    {"name": "Kavya Mishra", "class_12_percent": 85.7, "jee_percentile": 89.4, "fees": 50000},
    {"name": "Nikhil Raj", "class_12_percent": 79.1, "jee_percentile": 84.6, "fees": 70000},
    {"name": "Pooja Sahu", "class_12_percent": 90.5, "jee_percentile": 93.9, "fees": 35000},
    {"name": "Deepak Meena", "class_12_percent": 67.3, "jee_percentile": 71.2, "fees": 120000},
    {"name": "Ishita Rao", "class_12_percent": 92.8, "jee_percentile": 96.4, "fees": 25000},
    {"name": "Sahil Khan", "class_12_percent": 74.6, "jee_percentile": 80.5, "fees": 90000},
    {"name": "Mehak Agrawal", "class_12_percent": 87.4, "jee_percentile": 91.7, "fees": 50000},
    {"name": "Yash Tiwari", "class_12_percent": 82.9, "jee_percentile": 87.2, "fees": 50000},
    {"name": "Tanya Chouhan", "class_12_percent": 95.1, "jee_percentile": 99.0, "fees": 15000},
    {"name": "Arjun Malviya", "class_12_percent": 78.3, "jee_percentile": 83.8, "fees": 70000},
    {"name": "Divya Lodhi", "class_12_percent": 86.1, "jee_percentile": 90.8, "fees": 50000},
    {"name": "Kartik Bansal", "class_12_percent": 71.9, "jee_percentile": 77.6, "fees": 120000},
    {"name": "Nidhi Patel", "class_12_percent": 90.0, "jee_percentile": 94.2, "fees": 35000},
    {"name": "Lakshya Singh", "class_12_percent": 83.5, "jee_percentile": 88.1, "fees": 50000},
    {"name": "Muskan Ali", "class_12_percent": 88.9, "jee_percentile": 92.7, "fees": 35000}
]
df=pd.DataFrame(data)
#print(data)
scaler=MinMaxScaler()
convert=scaler.fit_transform(df[["class_12_percent","jee_percentile"]])

X=convert
y=df["fees"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
model=LinearRegression()
model.fit(X_train,y_train)
model.predict(X_test)
score=model.score(X_test,y_test)
#print(score)
       ##New User prediction
print("New Student fees Predict")
new_student_name=input("ENTER STUDENT NAME :")
new_student_12_per=float(input("ENTER YOUR CLASS 12 PERCENT :"))
new_student_jee_per=float(input("ENTER YOUR JEE PERCENTILE :"))
new_df=pd.DataFrame([[new_student_12_per,new_student_jee_per]],columns=["class_12_percent","jee_percentile"])
new_scaled=scaler.transform(new_df)
new_X=new_scaled
new_prediction=model.predict(new_X)
print(f"Student fees is {float(new_prediction[0])}")









