import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import sympy as sp
import scipy as sc
from scipy import integrate

def task1():
    # Создание матрицы 5x5 случайных целых чисел от 0 до 9
    matrix = np.random.randint(0, 10, (5, 5))

    print("Исходная матрица:")
    print(matrix)

    # Транспонирование матрицы
    transposed_matrix = np.transpose(matrix)

    print("\nТранспонированная матрица:")
    print(transposed_matrix)

    # Вычисление определителя транспонированной матрицы
    determinant = np.linalg.det(transposed_matrix)

    print(f"\nОпределитель транспонированной матрицы: {determinant}")


def task2():

    # Создание вектор-столбца (3x1)
    vector_column = np.array([[1], [2], [3]])

    # Создание матрицы (3x3)
    matrix = np.array([[4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]])

    # Умножение матрицы на вектор
    result = np.dot(matrix, vector_column)

    print("Вектор-столбец:")
    print(vector_column)

    print("\nМатрица:")
    print(matrix)

    print("\nРезультат умножения матрицы на вектор:")
    print(result)


def task3():
    x, y = sp.symbols('x y')
    expr = (2 * x - 3 * y) ** 2 - 4 * x * y * (x - y) / 3
    simplified_expr = sp.simplify(expr)
    value = simplified_expr.subs({x: 1.038, y: 7})
    print(value)




def task4():
    # Создание символьных переменных x и y
    x, y = sp.symbols('x y')

    # Задание выражения
    expression = (2 * x - 3 * y) ** 2 - 4 * x * y * (x - y) / 3

    # Нахождение частных производных
    partial_derivative_x = sp.diff(expression, x)
    partial_derivative_y = sp.diff(expression, y)

    print("Частная производная по x:")
    print(partial_derivative_x)

    print("\nЧастная производная по y:")
    print(partial_derivative_y)



def task5():
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    eq1 = x1 - x3 - 1
    eq2 = -x1 -x2 + 3*x3 + 3
    eq3 = x1 - 2*x2 - 4*x3 - 5
    solution_sympy = sp.solve((eq1, eq2, eq3), (x1, x2, x3))
    print(solution_sympy)

    coefficients = np.array([[1, 0, -1], [-1, -1, 3], [1, -2, -4]])
    constants = np.array([1, -3, 5])
    solution_numpy = np.linalg.solve(coefficients, constants)
    print(solution_numpy)


def task6():
    # Вычисляем интеграл численно
    result_scipy, _ = sc.integrate.quad(lambda x: x ** (1/2) + x ** (2/3), 0, 1)
    print("Интеграл с помощью SciPy:", result_scipy)

    # Создаем символьную переменную
    x = sp.symbols('x')
    # Вычисляем интеграл символьно
    result_sympy = sp.integrate(x ** (1/2) + x ** (2/3), (x, 0, 1))
    print("Интеграл с помощью SymPy:", result_sympy)

def task7():
    def f(x,y):
        return (x - y)*math.exp(1)**y
    def h(x):
        return x
    def g(x):
        return 2*x
    v, err = integrate.dblquad(f,-1,1,g,h)
    print(v)




def task8():
    plt.figure(figsize = (8, 5), dpi = 80)
    ax = plt.subplot(111)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    X = np.linspace(-5, 2 * np.pi, 512, endpoint=True)
    C, L = 3 * np.sin(X) , np.sqrt(X + 5)

    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="Sin Function")
    plt.plot(X, L, color="red", linewidth=2.5, linestyle="-", label="Sqrt Function")

    plt.xlim(X.min() * 1.1, X.max() * 1.1)
    plt.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+3 \pi/2$',
                r'$+ 2\pi$'])
    ax.set_xlabel("Y", fontsize=15, color='black', labelpad=-190)  # +
    ax.set_ylabel("X", fontsize=15, color='black', labelpad=-270, rotation = 0)



    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Добавление точки пересечения
    idx = np.argwhere(np.diff(np.sign(C - L))).flatten()
    plt.plot(X[idx], C[idx], 'o', color = 'orange')



    plt.legend(loc='upper left', frameon=False)
    plt.grid()
    plt.show()

print("Задание 1")
task1()
print("Задание 2")
task2()
print("Задание 3")
task3()
print("Задание 4")
task4()
print("Задание 5")
task5()
print("Задание 6")
task6()
print("Задание 7")
task7()
print("Задание 8")
task8()