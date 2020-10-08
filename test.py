from sklearn import linear_model
import numpy as np

x_input = [69, 67, 71, 65, 72, 68, 74, 65, 66, 72]
x_input = np.array(x_input).reshape(-1, 1)
y_input = [9.5, 8.5, 11.5, 10.5, 11, 7.5, 12, 7, 7.5, 13]
x_test = [70]
x_test = np.array(x_test).reshape(-1, 1)


reg = linear_model.LinearRegression()
reg.fit(x_input, y_input)
Y_pred = reg.predict(x_test)

print("Y_pred: ", Y_pred)
print("Coef: ", reg.coef_)