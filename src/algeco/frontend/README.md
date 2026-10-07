# Algeco frontend

The frontend's job is to provide a way to build the expressions we're interested in in Python, and dump them to the _internal representation_ the compiler will use for optimization.

- `expr.py` defines the constructs of the source language, with operator overloading to support a simple DSL:
```python
# define x with shape <4x4>, y with shape <4x8>
x, y = Var("x", (4, 4)), Var("y", (4, 8))

# define z as the result of matmul(x, y)
z = x @ y
# define t as the sum of z with itself
t = z + z
``` 
These do not _compute_ the results, they just build python classes that represent the computation we want to perform. This is a first form of in-memory IR, which we just directly convert to the xDSL IR.

- `jit.py` is an end-to-end wrapper for the full compiler. Annotating a function with `algeco.jit` marks it as JIT compiled. The function is not compiled immediately but on the first call, since that's when the shapes of the buffers are known. The result of the compilation is cached so that it's reused as long as the weights and shapes don't change.