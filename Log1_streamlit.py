import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.title('Model Deployment: Titanic Passenger Survival Probability Dashboard')

st.sidebar.header('Provide Passenger Details')


def Passenger_profile():
    Pclass = st.sidebar.selectbox('Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)', ('1', '2', '3'))
    Sex = st.sidebar.selectbox('Gender(1=male,0=female)', ('1', '0'))
    Age = st.sidebar.number_input("Enter the Age")
    SibSp = st.sidebar.selectbox('Number of Siblings/Spouses Aboard', ('0', '1', '2', '3', '4', '5', '8'))
    Parch = st.sidebar.selectbox('Number of Parent/children Aboard', ('0', '1', '2', '3', '4', '5', '6'))
    Fare = st.sidebar.number_input("Enter the Fare (in £)")
    data = {"Pclass": Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = Passenger_profile()
st.subheader('Passenger Details')
st.write(df)

Titanic = pd.read_csv("Titanic_train.csv")
Titanic["Age"].fillna(Titanic["Age"].median(), inplace=True)
Titanic.drop(columns=["PassengerId", "Name", "Ticket", "Cabin","Embarked"], inplace=True, axis=1)
Titanic = Titanic.dropna()
Titanic['Sex'] = Titanic['Sex'].map({'female': 0, 'male': 1})


X = Titanic.drop(columns=["Survived"])
Y = Titanic["Survived"]
clf = LogisticRegression()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Predicted Result:')
st.write('This person is survived' if prediction_proba[0][1] > 0.5 else 'This person is not survived')

st.subheader('Survival Probability Bar:')
def plot_survival_prob(probability):
    fig, ax = plt.subplots()
    ax.bar(['Survived', 'Not Survived'], [probability, 1 - probability], color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
plot_survival_prob(prediction_proba[0][1])

st.subheader('Prediction Probability:')
st.write("(0-Probability of Not Survived,1- Probability of Survived)")
st.write(prediction_proba)

