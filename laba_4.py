import numpy as np


def simplex_method(a, b, c):
    # Получение размеров матрицы A и вектора c
    m, n = len(a), len(c)

    # Добавление искусственных переменных
    for i in range(m):
        a[i].extend([0] * m)
        a[i][n + i] = 1
        a[i].append(b[i])
    c.extend([0] * (m + 1))

    xb_indices = list(range(n, n + m))  # Индексы базисных переменных
    dual_index = xb_indices

    while max(c[:-1]) > 0:
        pivot_col = c.index(max(c[:-1]))
        ratios = []
        for i in range(m):
            if a[i][pivot_col] == 0:
                ratios.append(float('inf'))
            else:
                if (a[i][-1] / a[i][pivot_col]) > 0:
                    ratios.append(a[i][-1] / a[i][pivot_col])

        pivot_row = ratios.index(min(ratios))  # строка с min ratios
        pivot_value = a[pivot_row][pivot_col]

        pivot_row_values = [(x / pivot_value) for x in a[pivot_row]]

        # обновляем таблицу
        for i in range(m):
            mem = a[i][pivot_col]
            for j in range(n + m + 1):
                if i == pivot_row:
                    a[i][j] = pivot_row_values[j]
                else:
                    a[i][j] -= pivot_row_values[j] * mem

        # обновляем вектор c
        c_max_value = c[pivot_col]
        for j in range(n + m + 1):
            if j == pivot_col:
                c[j] = 0
            elif j == m + n:
                c[j] += a[pivot_row][j] * c_max_value
            else:
                c[j] -= a[pivot_row][j] * c_max_value

        xb_indices[pivot_row] = pivot_col

    x = [0] * (n)
    for i in range(m):
        if xb_indices[i] < n:
            x[xb_indices[i]] = a[i][-1]

    return [c[-1], x]


# Генерируем случайную матрицу размером 6x8
# A = np.random.randint(1, 11, size=(8, 6))

A = np.array([[2, 4, 2, 3, 4, 6],
              [4, 6, 5, 1, 8, 2],
              [2, 5, 4, 3, 9, 8],
              [5, 2, 3, 2, 6, 5],
              [9, 7, 8, 2, 5, 5],
              [6, 3, 2, 8, 7, 4],
              [3, 9, 4, 3, 2, 9],
              [7, 4, 5, 4, 2, 3]])

n = len(A)
m = len(A[0])

A_tr = A.T
A = A.tolist()
A_tr = A_tr.tolist()

# Находим верхнюю и нижнюю цену игры
upper_price = np.min(np.max(A_tr, axis=1))
lower_price = np.max(np.min(A, axis=0))

# Находим равновесное решение в смешанных стратегиях используя симплекс-метод

b = [1] * n
c = [1] * m

b_y = [1] * m
c_y = [1] * n

x = simplex_method(A, b, c)
y = simplex_method(A_tr, b_y, c_y)

for i in range(len(x[1])):
    x[1][i] /= x[0]

for i in range(len(y[1])):
    y[1][i] /= y[0]

print('Смешанная стратегия x: ', x[1])
print('Смешанная стратегия y: ', y[1])

print('Верхняя цена игры: ', upper_price)
print('Нижняя цена игры: ', lower_price)
