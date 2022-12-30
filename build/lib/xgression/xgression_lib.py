from __future__ import annotations
from copy import copy, deepcopy
from typing import Any, Dict
import numpy as np
import sympy as sp
import scipy
import random as rd
import threading as th
import secrets
import math
import numbers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from contextlib import contextmanager
import threading
import _thread
import collections.abc
import warnings
warnings.filterwarnings("ignore")

optimization_method = 'BFGS'

INITIAL_RANDOM_FACTOR = 0
RANDOM_FACTOR_INCREMENT_FACTOR = 0.01
MAX_RANDOM_FACTOR = 0.7

global random_factor
random_factor = INITIAL_RANDOM_FACTOR

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg='calculation too long'):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

class Node:
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

        if self.a is not None:
            self.a.next = self
        if self.b is not None:
            self.b.next = self

        self.next = None
        self.name = None

        self.deleted = False

    def clone(self):
        root = Node(None,None)
        root.__class__ = self.__class__

        if not isinstance(self.a,Variable):
            root.a = self.a.clone()
        else:
            root.a = Variable(self.a.a,name=self.a.name)
        root.a.next = root
        
        if not isinstance(self.b,Variable):
            root.b = self.b.clone()
            #print(self.b)
        else:
            root.b = Variable(self.b.b,name=self.b.name)
        root.b.next = root
        #print(root)
        return root

    def replace_unaffected(self,remove_a):
        raise NotImplementedError

    def get_head(self):
        if self.next is None:
            return self
        return self.next.get_head()

    def get_all_nodes(self):
        if isinstance(self.a, Node):
            node_a = self.a.get_all_nodes()
        else:
            node_a = []
        
        if isinstance(self.b, Node):
            node_b = self.b.get_all_nodes()
        elif self.b is None:
            node_b = []
        else:
            node_b = []

        return [self] + node_a + node_b

    def optimize(self):
        # if self.a.eval() is number
        if isinstance(self.a.eval(), numbers.Number):
            self.a = Variable(self.a.eval())
        
        if isinstance(self.b.eval(), numbers.Number):
            self.b = Variable(self.b.eval())
        
        if isinstance(self.a, Node):
            self.a.optimize()
        
        if isinstance(self.b, Node):
            self.b.optimize()
    
    def eval(self):
        raise NotImplementedError

    def __call__(self):
        return self.eval()

    def delete(self,keep_a=True):
        """
        Deletes the current node from the tree
        """
        v = self.a if keep_a else self.b
        if self.next is not None:
            if self is self.next.a:
                self.next.a = v
            elif self is self.next.b:
                self.next.b = v
        v.next = self.next
        self.deleted = True

    def add_node_after(self, new_node:Node) -> Node:
        """
        Adds a new node after the current node

        Parameters
        ----------
        new_node : Node
            The new node to add
        
        Returns
        -------
        Node
            The new added node
        """
        new_node.a = self
        if self.next is not None:
            if self.next.a is self:
                new_node.a = self
                new_node.next = self.next
                self.next.a = new_node
            else:
                new_node.a = self
                new_node.next = self.next
                self.next.b = new_node
        else:
            new_node.a = self
        self.next = new_node
        return new_node

class Add(Node):
    def eval(self,return_variable=False):
        return self.a.eval(return_variable) + self.b.eval(return_variable)

    def __str__(self) -> str:
        return f'({self.a} + {self.b})'

    def replace_unaffected(self, remove_a):
        return 0

class Variable(Node):
    def __init__(self, a, name=None, b=None, next_node = None) -> None:
        self.a = a
        if isinstance(self.a,Variable):
            raise TypeError("Incorrect variable value")
        self.b = None
        self.name = name

        self.next = next_node
    
    def eval(self,return_variable=False):
        if not return_variable or self.name is not None:
            return self.a
        else:
            return sp.symbols(str(self.a))
    
    def __str__(self):
        if self.name is not None:
            return str(self.name)
        return str(self.a)

    def optimize(self):
        pass

    def replace_unaffected(self,remove_a):
        print("THIS IS ERROR")
    
class Sub(Node):
    def eval(self,return_variable=False):
        return self.a.eval(return_variable) - self.b.eval(return_variable)
    
    def __str__(self) -> str:
        return f'({self.a} - {self.b})'

    def replace_unaffected(self, replace_a):
        if replace_a:
            return 1
        else:
            return 0

class Mul(Node):
    def eval(self,return_variable=False):
        return self.a.eval(return_variable) * self.b.eval(return_variable)
    
    def __str__(self) -> str:
        return f'({self.a} * {self.b})'

    def replace_unaffected(self, remove_a):
        return 1

class Div(Node):
    def eval(self,return_variable=False):
        return self.a.eval(return_variable) / self.b.eval(return_variable)

    def __str__(self) -> str:
        return f'({self.a} / {self.b})'

    def replace_unaffected(self, remove_a):
        if not remove_a:
            return 1
        else:
            return 0

class Pow(Node):
    def eval(self,return_variable=False):
        try:
            return self.a.eval(return_variable) ** self.b.eval(return_variable)
        except:
            return 9999999999999999

    def __str__(self) -> str:
        return f'({self.a} ** {self.b})'

    def replace_unaffected(self, remove_a):
        return 1

class InvPow(Node):
    def eval(self,return_variable=False):
        try:
            return self.b.eval(return_variable) ** self.a.eval(return_variable)
        except:
            return 9999999999999999

    def __str__(self) -> str:
        return f'({self.b} ** {self.a})'

    def replace_unaffected(self, remove_a):
        return 1

class FunctionNode(Node):
    def __init__(self, a: Node, *args):
        self.a = a
        self.b = None

        self.a.next = self

        self.next = None
        self.name = None

        self.deleted = False
    
    def __str__(self) -> str:
        raise NotImplementedError
    
    def replace_unaffected(self, remove_a):
        raise NotImplementedError

class Sin(FunctionNode):
    def eval(self,return_variable=False):
        try:
            return np.sin(self.a.eval(return_variable))
        except:
            #print("ASDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
            #print(self.a.eval())
            return 9999999999999999999999999
    
    def __str__(self) -> str:
        return f'sin({self.a})'

    def replace_unaffected(self, remove_a):
        return 0

class Cos(FunctionNode):
    def eval(self,return_variable=False):
        try:
            return np.cos(self.a.eval(return_variable))
        except:
            #print("ASDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
            #print(self.a.eval())
            return 9999999999999999999999999
    
    def __str__(self) -> str:
        return f'cos({self.a})'

    def replace_unaffected(self, remove_a):
        return 0

class Tan(FunctionNode):
    def eval(self,return_variable=False):
        try:
            return np.tan(self.a.eval(return_variable))
        except:
            #print("ASDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
            #print(self.a.eval())
            return 9999999999999999999999999
    
    def __str__(self) -> str:
        return f'tan({self.a})'

    def replace_unaffected(self, remove_a):
        return 0

class Abs(FunctionNode):
    def eval(self,return_variable=False):
        return np.abs(self.a.eval(return_variable))
    
    def __str__(self) -> str:
        return f'abs({self.a})'

    def replace_unaffected(self, remove_a):
        return 0

class UnknownVariable(Variable):
    def __init__(self) -> None:
        self.a = sp.symbols("sfi292df")
        self.b = None
        self.name = "sfi292df"
        self.used_to_be_unknown = True

        super().__init__(self.a, self.name, self.b)

class UnknownOperator(Node):
    def __init__(self) -> None:
        self.a = None
        self.b = UnknownVariable()
        self.used_to_be_unknown = True

        #self.a.next = self
        self.b.next = self

        self.next = None
        self.name = None

    def eval(self,return_variable=False):
        return 0

    def __str__(self) -> str:
        return "Unknown"

    def best_value_with_the_node(self, tree:Xgression):
        best_value_name = None
        best_value = None
        best_dist = 99999999999
        best_operator = None
        best_prev = None
        ops = [Pow, Add, Sub, Mul, Div, Sin, Cos, Tan, Abs, InvPow]
        rd.shuffle(ops)
        for opr in ops:
            self.__class__ = opr
            if isinstance(opr,FunctionNode):
                dist = abs(tree.get_distance().mean())
                if dist < best_dist:
                    best_value = self.b.a
                    best_value_name = x_name
                    best_dist = dist
                    best_operator = opr
                    best_prev = self.a
                    if rd.random() < random_factor or dist == 0:
                        self.delete()
                        return best_value,best_dist,best_value_name,best_operator,best_prev
                self.delete()
                continue
            for x_name in tree.x:
                value_b = tree.x[x_name]
                self.b = Variable(value_b,x_name, next_node=self)
                self.b.next = self
                dist = abs(tree.get_distance().mean())
                #print(self.get_head(),dist)
                if dist < best_dist:
                    best_value = self.b.a
                    best_value_name = x_name
                    best_dist = dist
                    best_operator = opr
                    best_prev = self.a
                    if rd.random() < random_factor or dist == 0:
                        self.delete()
                        return best_value,best_dist,best_value_name,best_operator,best_prev
            
            #print(opr)
            self.b = UnknownVariable()
            self.b.next = self
            #print(self.get_head())
            #print("SADD",tree.find_variable_value_minimize_distant())
            minimize_value = tree.find_variable_value_minimize_distant()
            #print(minimize_value)
            self.b = Variable(minimize_value,str(minimize_value))

            #self.b = Variable(minimize_value,str(minimize_value))
            try:
                dist = abs(tree.get_distance().mean())
                #print(dist,opr,self,minimize_value)
            except:
                self.delete()
                return None,9999999999999999999999999,None,None,None
            if minimize_value is not None and dist < best_dist:
                best_value = minimize_value
                best_value_name = None
                best_dist = dist
                best_operator = opr
                best_prev = self.a

        self.delete()
        #print(best_value,best_dist,best_value_name,best_operator,best_prev)
        return best_value,best_dist,best_value_name,best_operator,best_prev
            
class Xgression:
    def __init__(self, x: Dict, y, y_name = None, classification = False) -> None:
        self.x = {xk : np.asarray(x[xk]) for xk in x}
        self.y = np.asarray(y)
        self.error_log = []
        self.best_eq = None
        self.best_error = 99999999
        self.classification = classification
        self.activation = lambda x:x
        if classification:
            def sigmoid_array(x):                                        
                return 1 / (1 + math.e**-x)
            self.activation = sigmoid_array

        if y_name is None:
            self.annotation = f"f({','.join(tuple(self.x.keys()))})"
        else:
            self.annotation = y_name
        #self.tree = Add(Variable(x["a"],name="a"),Variable(x["b"],name="b"))
        #vf = list(self.x.keys())[0]
        self.init_tree(x)
        self.ops = {"mo":self.find_operator_type_mutate,
                    "del":self.best_delete,
                    "add":self.find_the_best_way_to_add_node}
        self.ops_backup = copy(self.ops)

        fig,axs = plt.subplots(len(self.x))
        self.fig = fig
        try:
            self.axs = list(axs)
        except:
            self.axs = [axs]
        #if not isinstance(self.axs, collections.abc.Sequence):
        #    self.axs = [self.axs]
        self.loss_fn = self.mse
        print(self.axs,self.fig)

    def init_tree(self, x):
        for i,vf in enumerate(self.x):
            if i == 0:
                self.tree = Add(Variable(x[vf],name=vf),Variable(0,str(0)))
            else:
                self.tree = Add(self.tree,Variable(x[vf],name=vf))

    def rotate_head(self):
        self.tree = self.tree.get_head()

    def get_distance(self,with_symbol=False):
        return (self.y - self.get_answer(with_symbol=with_symbol))**2

    def get_answer(self,with_symbol=False):
        return self.activation(self.tree.get_head().eval(with_symbol))

    def execute_algorithm(self):
        # random order
        nodes = self.tree.get_all_nodes()
        current_dist = self.get_distance().mean()
        np.random.shuffle(nodes)
        next_node = None
        old_node = None
        solution_type = 0
        for node in nodes:
            if isinstance(node, Variable):
                continue
                dist, new_node, old_node = self.mutate_variable_node(node)
                if dist < current_dist:
                    solution_type = 2
                    current_dist = dist
                    next_value = new_node.a

            else:
                dist, next_class, old_node = self.operator_type_mutate(node)
            
                if dist >= current_dist and next_class:
                    #print("found better solution")
                    #print(next_class)
                    solution_type = 1
                    current_dist = dist
                    next_class = next_class
                    next_node = node

        if solution_type == 1 and next_class is not None:
            #print(next_node.__class__)
            next_node .__class__ = next_class

        elif solution_type == 2:
            return
            print(old_node)
            old_node.a = next_value
        
        return current_dist

    def operator_type_mutate(self, node):
        dist = self.get_distance().mean()
        old_class = deepcopy(node.__class__)
        next_class = None
        # random order [Add,Sub,Mul,Div,Pow,Neg]
        ot = [Add,Sub,Mul,Div,Pow]
        rd.shuffle(ot)
        rti = False
        if rd.random() < 0.1:
            rti = True
        for operator in ot:
            # replace node with new operator
            node.__class__ = operator
            try:
                next_dist = self.get_distance().mean()
                node.__class__ = old_class
            except Exception as e:
                #print(e)
                node.__class__ = old_class
                continue
                # print(next_dist,operator)
            node.__class__ = old_class
            if rti or next_dist < dist:
                dist = next_dist
                next_class = operator
                #node.__class__ = old_class
                return dist,next_class,node
                print(next_class,self.tree)
                    
            node.__class__ = old_class
        return dist,next_class,node

    def find_operator_type_mutate(self):
        best_dist = 999999999
        the_class = None
        next_node = None
        for node in self.tree.get_all_nodes():
            if not isinstance(node,Variable):
                dist,next_class,node = self.operator_type_mutate(node)
                if next_class is not None and dist < best_dist:
                    next_node = node
                    the_class = next_class
                    best_dist = dist
        return best_dist,the_class,next_node

    def find_variable_value_minimize_distant2(self):
        try:
            with time_limit(10):
                dist = self.get_distance()
                max_len = len(dist)
                #dist = rd.sample(dist,max_len if max_len <= 20 else 20)
                mean_answer = 0
                for i in range(max_len):
                    try:
                        mean_answer += sp.solve(dist[i],quick=True,rational=False,simplify=False)[0]
                    except Exception as e:
                        max_len-=1
                        pass
                if mean_answer == 0:
                    return
                mean_answer = mean_answer/max_len
                return mean_answer
        except BaseException as e:
            return None
            pass

    def find_variable_value_minimize_distant(self):
        with time_limit(10):
            dist = self.mse()
            #print(dist)
            #x0 = np.random.random(len(dist)).tolist()
            try:
                
                f = sp.lambdify(sp.Symbol("sfi292df"), dist, modules='numpy')
                mv = scipy.optimize.minimize(f,1,method=optimization_method).x[0]
                #print(type(mv),"PASSED")
                return mv
            except Exception as e:
                #print(e)
                return None
                #return self.find_variable_value_minimize_distant2()
                

    def mse(self,with_symbol=False):
        return abs(self.get_distance(with_symbol).mean())
    
    def find_the_best_way_to_add_node(self,best_dist=9999999999999999):
        best_node : Node = None
        prev_node : Node = None
        best_value = None
        best_value_name = None
        best_operator = None

        nodes = self.tree.get_all_nodes()
        rd.shuffle(nodes)
        for node in nodes:
            new_node = UnknownOperator()
            new_node : UnknownOperator = node.add_node_after(new_node)
            value,dist,name,operator,_ = new_node.best_value_with_the_node(self)
            if dist < best_dist:
                best_dist = dist
                best_operator = operator
                prev_node = node
                #print(node,dist,node.get_head())
                best_value = value
                best_value_name = name
                if rd.random() < random_factor:
                    return best_dist,prev_node,best_operator,best_value,best_value_name
        if best_value is None:
            return 999999999999999999,None,None,None,None
        return best_dist,prev_node,best_operator,best_value,best_value_name

    def best_delete(self):
        best_dist = 9999999999
        best_node = None
        del_a = None
        for node in self.tree.get_all_nodes():
            if not isinstance(node, Variable):
                # try remove
                for is_a in [True,False]:
                    replace_value = node.replace_unaffected(is_a)
                    if is_a:
                        old_a = copy(node.a)
                        node.a = Variable(replace_value)
                        dist = self.mse()
                        node.a = old_a
                    else:
                        old_b = copy(node.b)
                        node.b = Variable(replace_value)
                        dist = self.mse()
                        node.b = old_b

                    if dist < best_dist:
                        best_dist = dist
                        best_node = node
                        del_a = is_a
        return best_dist,best_node,del_a

    def get_op(self):
        op = None
        type_opr = None
        new_dist = 9999
        opl = list(self.ops.keys())
        rd.shuffle(opl)
        for opr in opl:
            try:
                ophd = self.ops[opr]()
            except:
                continue
            if ophd[0] < new_dist:
                type_opr = opr
                op = ophd
                new_dist = ophd[0]
                #print(op)
                if rd.random() < random_factor-0.5 or opr == "add":
                    return new_dist,op,type_opr
        return new_dist,op,type_opr

    def get_variables(self):
        nodes = [node for node in self.tree.get_head().get_all_nodes() if isinstance(node,Variable) and node.name is None]
        #print("DBG")
        return nodes


    def iteration(self):
        """
        ways = {}
        try:
            pass
            ways["1del"] = list(self.best_delete())
        except:
            pass
        try:
            ways["2add"] = list(self.find_the_best_way_to_add_node())
        except:
            pass
        try:
            pass
            ways["3mo"] = list(self.find_operator_type_mutate())
        except:
            pass
        """

        new_dist,op,type_opr = self.get_op()

        current_error = self.mse()
        #print(list(ways.items())[0])
        #type_opr = op[0]
        #op = tuple(op[1])
        #print(op)
        #print(type_opr)
        #new_dist = list(ways.items())[0][0]
        try:
            if new_dist < current_error or rd.random() < 1:
                self.error_log.append(new_dist)
                if type_opr == "add":
                    _,prev_node,best_operator,best_value,best_value_name = op
                    prev_node.add_node_after(best_operator(Variable("gae2a2"),Variable(best_value,name=best_value_name)))
                    #self.tree = prev_node.get_head()
                elif type_opr == "del":
                    #print("DEL",op[1],op[1].replace_unaffected(op[2]))
                    #print(op[1].a,op[1].b)
                    _,best_node,del_a = op
                    if best_node is not None:
                        value = Variable(best_node.replace_unaffected(del_a))
                        if del_a:
                            best_node.a = value
                            #self.tree = best_node.a.get_head()
                        else:
                            best_node.b = value
                            #self.tree = best_node.a.get_head()
                elif type_opr == "mo":
                    #print("MO",op[1])
                    _,the_class,next_node = op
                    #print(the_class,new_dist)
                    next_node.__class__ = the_class
            #self.tree = Add(Variable(self.x["a"],name="a"),Variable(self.x["b"],name="b"))
            #print(new_dist)
            self.rotate_head()
            #self.optimize_tree()
            ce = float(self.mse())
            if ce < self.best_error:
                self.best_error = ce
                self.best_eq = sp.simplify(str(self.tree))
                global random_factor
                random_factor = INITIAL_RANDOM_FACTOR
            else:
                if random_factor >= MAX_RANDOM_FACTOR:
                    #random_factor = MAX_RANDOM_FACTOR
                    random_factor = INITIAL_RANDOM_FACTOR
                    self.init_tree()
                else:
                    random_factor += RANDOM_FACTOR_INCREMENT_FACTOR
            self.optimize_all_tree_variable()
        except:
            return 99999999999999999
        return self.mse()
                        
    def optimize_tree(self):
        self.rotate_head()
        self.tree.optimize()

    def optimize_all_tree_variable(self):
        try:
            #ev = t.tree.eval(return_variable=True)
            dist = self.mse(with_symbol=True)
            symbols = list(dist.free_symbols)
            init = [(float(str(symbols[i]))) for i in range(len(symbols))]
            f = sp.lambdify(symbols, dist, modules='numpy')
            #init = list(np.random.random(len(symbols))*0+1)
            #print(f(*init))
            fn = lambda v : f(*v)
            mv = scipy.optimize.minimize(fn,init,method=optimization_method)
            if mv.success:
                mv = mv.x
                variables = self.get_variables()[::-1]
                #print(mv,symbols,[i.a for i in variables],"SDDD")
                for v_ref,v_value in zip(self.get_variables(),mv):
                    v_ref.a = v_value
            else:
                #print(mv)
                pass
        except Exception as e:
            print(e)
            pass

    def show_unpause(self,scatter=False):
        error = self.mse()
        self.fig.suptitle(f'MSE = {error}\n{self.get_equation()}')
        for ax,key in zip(self.axs,self.x):
            ax.cla()
            ax.set_title(f'{key} vs {self.annotation}')
            #ax.set_title(f'MSE = {dist}\n{sp.simplify(str((self.tree.get_head())))}')
            if not scatter:
                ax.plot(self.x[key],self.tree.get_head().eval(),c="r")
            else:
                ax.scatter(self.x[key],self.tree.get_head().eval(),c="r")
            ax.scatter(self.x[key],self.y,c="g")
            ax.set_xlabel(key)
            ax.set_ylabel(self.annotation)
        #print("SHOW")
        #fig.show()
        self.fig.tight_layout()
        plt.pause(0.01)
    
    def show_pause(self):
        error = self.mse()
        self.fig.suptitle(f'MSE = {error}\n{self.annotation} = {self.get_equation()}')
        for ax,key in zip(self.axs,self.x):
            ax.cla()
            ax.set_title(key)
            #ax.set_title(f'MSE = {dist}\n{sp.simplify(str((self.tree.get_head())))}')
            ax.plot(self.x[key],self.tree.get_head().eval(),c="r")
            ax.scatter(self.x[key],self.y,c="g")
            ax.set_xlabel(key)
            ax.set_ylabel(self.annotation)
        #print("SHOW")
        #fig.show()
        self.fig.tight_layout()
        plt.show()

    def mape(self):
        y_pred = self.tree.get_head().eval()
        y_true = self.y
        mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
        return mape

    def get_equation(self):
        if not self.classification:
            return f'{self.annotation} = {sp.simplify(str((self.tree.get_head())))}'
        return f'{self.annotation} = Sigmoid({sp.simplify(str((self.tree.get_head())))})'

if __name__ == "__main__":
    case = 3
    #rd.seed(20)

    if case == 1:
        a = np.linspace(0.1,8,50)
        b = np.linspace(-5,5,50)/a**3
        c = np.sin(np.linspace(-6,8,50))*3
        n = np.random.random(50)*3-1
        y_true = a**2+b*a*c+3*c+a**-2
        variables = {'a': a, 'b':b, 'c':c}

    if case == 2:
        a = np.linspace(0.1,8,50)
        b = np.linspace(-5,5,50)**2
        c = np.sin(np.linspace(-6,8,50))*1
        n = np.random.random(50)*3-1
        y_true = np.cos(a**2)+np.sin(b*a)+c
        variables = {'a': a, 'b':b, 'c':c}


    if case == 3:
        a = np.linspace(0.1,8,50)
        b = np.linspace(-5,5,50)**2
        c = np.sin(np.linspace(-6,8,50))*1
        n = (np.random.random(50)-0.5)*100
        y_true = (a**2+b*a+c)+n
        variables = {'a': a, 'b':b, 'c':c}


    if case == 4:
        a = np.linspace(0.1,8,50)
        b = np.linspace(-5,5,50)**2
        c = np.sin(np.linspace(-6,8,50))*1
        n = np.random.random(50)*3-1
        y_true = (a**2+b*a+c)
        variables = {'a': a, 'b':b, 'c':c}

    if case == 5:
        a = np.linspace(0.1,8,50)
        y_true = 2**(a-3)
        variables = {'a': a}

    if case == 6:
        a = np.linspace(0.1,2,50)
        y_true = a**4+4*a**3+a**2+a+3
        variables = {'a': a}


    t = Xgression(variables, y_true)
    #t.find_variable_value_minimize_distant2()


    
    #v = deepcopy(t)
    #t.tree = Add(Pow(Variable(a,name="a"),Variable(2)),Add(Variable(a,name="a"),Variable(5)))
    #r = t.find_the_best_way_to_add_node()
    #print(t.tree.get_head())

    #j = Add(Variable('a'),Variable('b'))
    #v = j.get_all_nodes()[2]
    #print(v)
    #v.add_node_after(Mul(Variable(66),Variable(8)))

    #print(j.get_head())
    bdt = 999999999999
    try:
        while True:
            dist = t.iteration()
            if dist <= 0.01:
                t.show_pause()
                break
            #print(t.tree)
            try:
                if dist < bdt:
                    bdt = dist
                    #print(t.optimize_all_tree_variable())
                    t.show_unpause(scatter=True)
            except Exception as e:
                print(e)
                pass
            #print(t.tree,float(t.mse()),t.best_mse,t.best_eq)
            #print(t.best_mse,random_factor,t.best_eq)
    except KeyboardInterrupt:
        pass

    print(t.tree.get_head(),float(t.mse()))

    pbd = 999999999999
    



    """
    for i in range(3000):
        best_dist,prev_node,best_operator,best_value,best_value_name = t.find_the_best_way_to_add_node()
        op_type = 1
        #print(best_dist)

        if prev_node is not None and (best_dist < pbd or rd.random() < 0.5):
            prev_node.add_node_after(best_operator(Variable("gae2a2"),Variable(best_value,name=best_value_name)))
            #t.optimize_tree()
            if best_dist < pbd:
                pbd = best_dist
                print(t.get_distance().mean())
                plt.clf()
                plt.title(f'MSE = {best_dist}\n{sp.simplify(str((t.tree.get_head())))}')
                plt.plot(t.x["a"],t.tree.get_head().eval(),c="r")
                plt.scatter(t.x["a"],t.y,c="g")
                plt.pause(0.1)
            pass
        if best_dist == 0:
            break
        t.rotate_head()
        #print(t.get_distance())

    print(t.tree.get_head(),t.tree.eval(),t.y)
    print(sp.simplify(str((t.tree.get_head()))),t.get_distance().mean())

    plt.clf()
    plt.title(f'MSE = {best_dist}\n{sp.simplify(str((t.tree.get_head())))}')
    plt.plot(t.x["a"],t.tree.get_head().eval(),c="r")
    plt.scatter(t.x["a"],t.y,c="g")
    plt.show()

    #print(best_dist,prev_node,best_operator,best_value,best_value_name)
    #print(t.find_variable_value_minimize_distant())
    """

    """
    for i in range(10000):
        dist = t.execute_algorithm()
        t.solve_equation()
        #print(dist,t.tree,t.solve_equation())
        if dist == 0:
            break
    """
    #print(t.tree,t.get_distance().mean())

    #g = Add(Variable(10),Mul(Variable(2),Variable(5)))
    #print(g.clone())