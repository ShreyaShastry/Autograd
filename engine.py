import math

class Value:
    def __init__(self,data,_children=(),_op="",label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children) #._prev means this is an internal attribute, not a public one. do not touch unless you know what you are doing.
        self._op = _op
        self.label = label
        self._backward = lambda:None


    #pretty print
    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self,other):
        # checking if other is already a Value. if it is a numeric value, then convert it into Value class too 
        other =  other if isinstance(other,Value) else Value(other)

        #out is the output Value after the operation
        out = Value(self.data + other.data, (self,other), _op = "+")  

        #_backward for only unique derivative rules, like mul,add and pow. sub and div can be derived from other op
        def _backward():
            self.grad += 1.0 * out.grad # self is x
            other.grad += 1.0 * out.grad # other is y
        out._backward = _backward
        return out
            

    def __sub__(self,other):
        return self + (-other)
    def __neg__(self):
        return self * -1


    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other): 
        return other + (-self)

    def __radd__(self, other): 
        return self + other
    def __rmul__(self, other): 
            return self * other
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self,other):
        # assert because Value*Value, meaning x^y, df/dy will give (x^y)lnx 
        assert isinstance(other,(int,float)) # only int and float supported. no variable power
        #other is strictly an integer/float
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad
        out._backward = _backward
        return out
        

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out
    def __truediv__(self,other):
        return self * other ** -1

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                visited.add(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

if __name__ == "__main__":
        x1 = Value(2.0, label = "x1")
        x2 = Value(0.0, label = "x2")

        w1 = Value(-3, label = "w1")
        w2 = Value(1, label = "w2")

        b = Value(6,label="b")

        x1w1 = x1 * w1
        x2w2 = x2 * w2
        x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1w1 + x2w2"

        n = x1w1x2w2 + b
        n.label="x1w1 + x2w2 + b"

        e = (2 * n).exp()
        e.label = 'e'

        num = e - 1
        den = e + 1
        num.label = "num"
        den.label = "den"

        o = num/den

        o.label = "o"

        o.backward()
        #print(e.grad)


        
    
    
        