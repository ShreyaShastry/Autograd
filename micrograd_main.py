from visualize import draw_dot as visualize
from engine import Value

# Inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8814, label="b")

# Compute weighted sums
x1w1 = x1 * w1
x1w1.label = "x1*w1"

x2w2 = x2 * w2
x2w2.label = "x2*w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"

n = x1w1x2w2 + b
n.label = "n"

# Intermediate: 2 * n
two_n = 2 * n
two_n.label = "2*n"

# Exponential
e = two_n.exp()
e.label = "e"

# Numerator and denominator
num = e - 1
num.label = "num"

den = e + 1
den.label = "den"

# Final output
o = num / den
o.label = "o"

# Backpropagate gradients
o.backward()

"""
print(f"x1.grad = {x1.grad}")
print(f"x2.grad = {x2.grad}")
print(f"w1.grad = {w1.grad}")
print(f"w2.grad = {w2.grad}")
print(f"b.grad = {b.grad}")
"""

# Visualize graph
dot = visualize(o)
dot.view()  
