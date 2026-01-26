
def _gradient(x, a, b):
    return (2*a*x + b)



def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    
    for _ in range(steps):

        x0 = x0 - lr*(_gradient(x0,a,b))

    return x0