from sympy import symbols, integrate


# Define variables
x1, x2, x3 = symbols('x1 x2 x3')

# Define the function to integrate
f = 16 * 3/8 * (2*x1 + 2*x2 - 1)*(2*x1 + 2*x2 -1)

# Set up the integral over the simplex
integral_result = integrate(
    integrate(
        integrate(f, (x1, 0, 1 - x2 - x3)),
    (x2, 0, 1 - x3)),
(x3, 0, 1))

print(integral_result)