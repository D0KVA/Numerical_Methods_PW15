import numpy as np
import matplotlib.pyplot as plt

def runge(f, x0, y0, h, x_end):
    n = int((x_end - x0) / h) + 1 
    x = np.linspace(x0, x_end, n)
    y = np.zeros(n)
    y[0] = y0
    
    for i in range(1, n):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h / 2, y[i-1] + k1 / 2)
        k3 = h * f(x[i-1] + h / 2, y[i-1] + k2 / 2)
        k4 = h * f(x[i-1] + h, y[i-1] + k3)
        
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x, y

def func(x, y):
    return y - 5*x

x0 = -1       
y0 = 1       
h = 1      
x_end = 1  

x_values, y_values = runge(func, x0, y0, h, x_end)

print("x", "\t", "y")
for i in range(len(x_values)):
    print(f"{x_values[i]:.1f}\t{y_values[i]:.4f}")

# График
plt.plot(x_values, y_values, 'o-', label="Метод Рунге-Кутты 4-го порядка")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Решение для y' = y - 5x используя Метод Рунге-Кутты 4-го порядка")
plt.legend()
plt.grid()
plt.show()