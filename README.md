# Xgression
## The Dynamic and Explainable Regression/Classification Solution
## Author : Thaphon Chinnakornsakul

Xgression is the solution to find relationships between data to an equation by constructing computational tree that represent the steps of calculation from inputs to an output of algorithm then modify the tree with various operator to find the optimal solution. As the result Xgression can perform many forms of regression without setting initial equations or parameters

Keywords : Theoretical Computer Science, Artificial Intelligence and Machine Learning, Computational Mathematics, Regression, Optimization, Algorithm, Explainable Machine Learning, Computational Tree, Predictive Modelling, Classification

DOI: https://doi.org/10.21203/rs.3.rs-2390968/v1

## installation
### via pip:
```
pip install Xgression
```

## Usage
```
from xgression.xgression_lib import Xgression
import numpy as np

## DUMMY data
u = np.array([1,2,3,4,5,6,7])
a = np.array([3,5,4,6,5,7,6])
s = a+3
v = np.sqrt(u**2+2*a*s)

inputs = {"u":u,"a":a,"s":s}

model = Xgression(inputs,v,y_name="v")

min_error = 999999999

while True:
    error = model.iteration()
    #print(error,model.get_equation())
    if error < min_error:
        min_error = error
        print(error,model.get_equation())
    if error <= 0.000001:
        break

### THE XGRESSION FOUND THIS SOLUTION (error = 0.09823555898800733)

# f(x1,X2) = ((cos(x1) - 0.9352702382439612)*(1.0087062458005631*1.4360224981242065**(x1**0.9982406583982952) 
# + X2**0.4728381686205922 + 2.1078702588835996) - 0.015435836069541794)/(cos(x1) - 0.9352702382439612)
```


## Reference:
[1]	Nocedal, Jorge; Wright, Stephen J. (2006), Numerical Optimization (2nd ed.), Berlin, New York: Springer-Verlag, ISBN 978-0-387-30303-1
[2]	Python Software Foundation. Python Language Reference, version 3.11. Available at http://www.python.org
[3]	Meurer A, Smith CP, Paprocki M, Čertík O, Kirpichev SB, Rocklin M, Kumar A, Ivanov S, Moore JK, Singh S, Rathnayake T, Vig S, Granger BE, Muller RP,
Bonazzi F, Gupta H, Vats S, Johansson F, Pedregosa F, Curry MJ, Terrel AR, Roučka Š, Saboo A, Fernando I, Kulal S, Cimrman R, Scopatz A. (2017) SymPy:
symbolic computing in Python. *PeerJ Computer Science* 3:e103https://doi.org/10.7717/peerj-cs.103
[4]	Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.4
[5]	Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
[6] Visit This [Predict Diabetes Kaggle](https://www.kaggle.com/datasets/whenamancodes/predict-diabities) for testing data

# Note; The Code is still Unclear and unrefactored. i just wrote what i think
# I will definitely refactor this after i pass the university interviewing!!
