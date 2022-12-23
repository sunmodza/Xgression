from xgression_lib import Xgression
import pandas as pd
import numpy as np
import string
alphabet = list(string.ascii_lowercase)


file_name = "diabetes.csv"
x_names = ["BloodPressure","Glucose","Insulin","BMI"]
y_name = "Outcome"
classification = False

data = pd.read_csv(file_name).sample(50).to_dict()
print(data.keys())
new = {}



for i,key in enumerate(x_names):
    new[key] = np.array(list(data[key].values()))


print(data["Outcome"])

y = np.array(list(data[y_name].values()))

model = Xgression(new,np.array(y),y_name=y_name,classification=classification)
for i in range(3000):
    model.iteration()
    try:
        model.show_unpause(scatter=True)
        print(model.mse())
        with open("previous_rel.txt","w") as log:
            log.write(model.get_equation())
    except:
        pass
