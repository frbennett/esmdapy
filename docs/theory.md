<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
# The Rosenbrock

Before we start to calibrate a real model, we start with some numerical optimization problems. They are all about finding the optima, just like a model. 
We start here with a simple two dimensional (parameter) function: The Rosenbrock (aka. Banana), which your might already know from the [Getting started chapter](../getting_started.md). 
The Rosenbrock function is defined as:

$$f_{Rosen}(x,y) = (100(y - x^2)^2 + (1-x)^2$$ 
 
where we defined control variables as  *-10 < x < 10* and *-10 < y < 10*, with *f(x=1,y=1) = 0* 