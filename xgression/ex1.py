from xgression_lib import Xgression
import numpy as np


x1 = np.array([1,2,3,4,5,6,7])
x2 = np.array([3,5,4,6,5,7,6])
y = np.array([5,6,7,9,11,13,17])

inputs = {"x1":x1,"X2":x2}

model = Xgression(inputs,y)

while True:
    error = model.iteration()
    print(error,model.get_equation())
    if error <= 0.1:
        break

### THE XGRESSION FOUND THIS SOLUTION (error = 0.09823555898800733)

# f(x1,X2) = ((cos(x1) - 0.9352702382439612)*(1.0087062458005631*1.4360224981242065**(x1**0.9982406583982952) 
# + X2**0.4728381686205922 + 2.1078702588835996) - 0.015435836069541794)/(cos(x1) - 0.9352702382439612)