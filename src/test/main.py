from __future__ import annotations
from typing import Dict, List, Self
import sympy as sp
import numpy as np
import random as rd


class Node:
    def __init__(self, value, prev=None, next=None):
        self.value = value
        self.next = None
        self.prev = None

    def travel_prev(self,ele=[]) -> List[Operator]:
        #ele.append(self.value)
        if self.prev is not None:
            return self.prev.travel_prev(ele=ele)
        else:
            return ele

    def get_value(self):
        if isinstance(self.value,Operator):
            return self.value.get_value()
        return self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

class Operator:
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

        self.a.next = self
        if b is not None:
            self.b.next = self

    def operation(self):
        raise NotImplementedError
    
    def __str__(self):
        return str(f'{self.a} + {self.b}')

    def travel_prev(self,ele=[]) -> List[Operator]:
        ele.append(self)
        if self.a is not None:
            ele = self.a.travel_prev(ele=ele)
            if self.b is not None:
                ele += self.b.travel_prev(ele=[])
            return ele
        else:
            ele[-1].border = True
            return ele

    def eval(self):
        a,b = self.get_value_of()
        value =  Node(self.operation(a,b))
        value.prev = self
        return value

    def get_value_of(self):
        if isinstance(self.a, Operator):
            a = self.a().value
        else:
            a = self.a.value
        
        # if b is Operator, eval it
        if isinstance(self.b, Operator):
            b = self.b()
        elif self.b is None:
            b = None
        else:
            b = self.b.value
        return a,b

    def __call__(self):
        return self.eval()

class Add(Operator):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)

    def operation(self,a,b):
        # if a is Operator, eval it
        return a + b

    def __str__(self):
        return str(f'({self.a} + {self.b})')

class Sub(Operator):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)

    def operation(self,a,b):
        return a - b

    def __str__(self):
        return str(f'({self.a} - {self.b})')

class Mul(Operator):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)

    def operation(self,a,b):
        return a * b

    def __str__(self):
        return str(f'({self.a} * {self.b})')

class Div(Operator):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)

    def operation(self,a,b):
        return a / b

    def __str__(self):
        return str(f'({self.a} / {self.b})')

class Pow(Operator):
    def __init__(self, a: Node, b: Node):
        super().__init__(a, b)

    def operation(self,a,b):
        return a**b

    def __str__(self):
        return str(f'({self.a} ** {self.b})')

class Neg(Operator):
    def __init__(self, a: Node):
        super().__init__(a, None)

    def operation(self,a,b):
        return -a
    
    def __str__(self):
        return str(f'-({self.a})')

class Xgression:
    def __init__(self, x:Dict, y):
        self.x = x
        self.y = y
        self.tree = Sub(Add(Node(0),Node(0)),Node(2))

        self.varaibles = {k:Node(self.x[k]) for k in self.x}

    def execute(self):
        for i in self.tree.travel_prev():
            # Assign Operator next
            # i.
            print(i.a.next)

        

t = Xgression(x={'a':1,'b':2},y=3).tree.travel_prev()[0]
print(t.b)
t.b.next = Sub(t.b,Node(5))
v = t.b.next
t.b = v

print(f'{t}')

'''
a = Neg(Sub(Add(Node(3),Node(2)),Node(2)))
ns = a().travel_prev()
print(ns)
#print(ns[-1])
ns[-1].b.value = 7

print(a())
print(f'{a} = {a()}')
'''
