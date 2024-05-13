import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def task1():
    matrix = np.random.randint(-3, 4, (10, 10))

    minor = matrix[1:5, 6:]

    determinant = np.linalg.det(minor)

    print("Исходная матрица:")
    print(matrix)
    print("\nМинор 4-го порядка:")
    print(minor)
    print("\nОпределитель минора 4-го порядка:")
    print(determinant)


def task2():
    vector = np.random.randint(-10, 11, size=(1, 10))

    def custom_norm(vector):
        norm_value = np.sqrt(np.sum(np.square(vector)))
        return norm_value

    custom_norm_result = custom_norm(vector)

    norm_np = np.linalg.norm(vector)

    print("Вектор-строка:")
    print(vector)
    print("\nНорма, вычисленная самостоятельно:", custom_norm_result)
    print("Норма, вычисленная с использованием linalg.norm():", norm_np)


def task3():
    matrix = np.random.randint(-10, 11, size=(5, 5))

    def custom_spectral_norm(matrix):
        # Вычисление собственных значений
        eigenvalues = np.linalg.eig(matrix)[0]

        # Спектральная норма - максимальный по модулю собственный значение
        spectral_norm = max(abs(eigenvalues))

        return spectral_norm

    # Вычисление спектральной нормы с помощью самописного алгоритма
    custom_spectral_norm_result = custom_spectral_norm(matrix)

    # Вычисление спектральной нормы с использованием linalg.norm()
    spectral_norm_np = np.linalg.norm(matrix, 2)

    # Вывод результатов
    print("Матрица:")
    print(matrix)
    print("\nСпектральная норма, вычисленная самостоятельно:", custom_spectral_norm_result)
    print("Спектральная норма, вычисленная с использованием linalg.norm():", spectral_norm_np)


def task4():
    # Представьте вашу систему уравнений в виде матрицы A и вектора b: A * x = b
    A = np.array([[4.4, -2.5, 19.2, -10.8], [5.5, -9.3, -14.2, 13.2], [7.1, -11.5, 5.3, -6.7], [14.2, 23.4, -8.8, 5.3], [8.2, -3.2, 14.2, 14.8]])
    b = np.array([4.3, 6.8, -1.8, 7.2, -8.4])

    # Находим псевдорешение с помощью метода наименьших квадратов
    x_pseudo = np.linalg.lstsq(A, b, rcond=None)[0]

    print("Псевдорешение системы:")
    print(x_pseudo)

def task5():
    points = np.array([[20, 15], [30, 40], [40, 21], [50, 65], [60, 54]])

    # Извлечение координат x и y из точек
    x = points[:, 0]
    y = points[:, 1]

    # Аппроксимация данных с помощью полинома второй степени (квадратичная функция)
    poly_coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(poly_coeffs)

    # Создание данных для построения аппроксимированной кривой
    x_curve = np.linspace(min(x), max(x), 100)
    y_curve = poly(x_curve)

    # Построение графика точек и аппроксимированной кривой
    plt.figure()
    plt.scatter(x, y, color='blue', label='Заданные точки')
    plt.plot(x_curve, y_curve, color='red', label='Аппроксимированная траектория')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация траектории')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод уравнения аппроксимированной траектории
    print("Уравнение наиболее выгодной траектории (квадратичная функция):")
    print(poly)


def task6():
    A = np.array([[4.4, -2.5, 19.2, -10.8], [5.5, -9.3, -14.2, 13.2], [7.1, -11.5, 5.3, -6.7], [14.2, 23.4, -8.8, 5.3]])
    b = np.array([4.3, 6.8, -1.8, 7.2])

    def decompose_to_LU(a):

        # create emtpy LU-matrix
        lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
        n = a.shape[0]

        for k in range(n):
            # calculate all residual k-row elements
            for j in range(k, n):
                lu_matrix[k, j] = a[k, j] - np.dot(lu_matrix[k, :k], lu_matrix[:k, j])
            # calculate all residual k-column elemetns
            for i in range(k + 1, n):
                lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

        return lu_matrix

    LU = decompose_to_LU(A)

    def solve_LU(lu_matrix, b):
        # get supporting vector y
        y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
        for i in range(y.shape[0]):
            y[i, 0] = b[i] - lu_matrix[i, :i] * y[:i]

        # get vector of answers x
        x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
        for i in range(1, x.shape[0] + 1):
            x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0]) / lu_matrix[-i, -i]

        return x

    x1 = solve_LU(LU, b)

    print("Решение системы X:")
    print(x1)

    x_check = np.linalg.solve(A, b)
    print("Проверка с помощью np.linalg.solve:")
    print(x_check)


task6()