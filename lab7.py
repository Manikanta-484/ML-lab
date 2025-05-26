import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data[["AveRooms"]], data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, c='b')
    plt.plot(X_test, y_pred, c='r')
    plt.title("Linear Regression - California Housing")
    plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

def polynomial_regression():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    df = pd.read_csv(url, sep='\s+', names=cols, na_values="?").dropna()
    X, y = df[["displacement"]], df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, c='b')
    plt.scatter(X_test, y_pred, c='r')
    plt.title("Polynomial Regression - Auto MPG")
    plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    linear_regression()
    polynomial_regression()
