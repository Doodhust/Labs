import numpy as np


#Значение функции.
def f(A, b, x):
    f_ = 0.5 * x.T @ A @ x + b.T @ x
    return f_


#Вычисляется якобиан системы уравнений.
def Jacobi_mat(A, x, x0, y):
    J = A + 2 * np.eye(4) * y
    J = np.pad(J, [(0, 1), (0, 1)], mode='constant', constant_values=0)

    tmp1 = 2 * (x - x0)
    tmp1 = np.pad(tmp1, (0, 1), mode='constant', constant_values=0)
    J[:, 4] = tmp1

    tmp2 = 2 * (x - x0).T
    tmp2 = np.pad(tmp2, (0, 1), mode='constant', constant_values=0)
    J[4, :] = tmp2

    J[4, 4] = 0

    return J


# Вектор f1
def L(A, x, x0, lamb, b, r):
    f1 = (A + 2 * np.eye(4) * lamb) @ x + (b + 2 * lamb * x0)
    f1 = np.pad(f1, (0, 1), mode='constant', constant_values=0)
    f1[4] = np.linalg.norm(x - x0) - r * r
    return f1


A = np.array([[2, 1, 5, -4],
              [1, 6, 3, -9],
              [5, 3, 8, 7],
              [-4, -9, 7, 3]])

print("A:")
print(A)
print()

b = np.array([2, 1, 0.3, 6])
print("b:")
print(b)
print()

x0 = np.array([1, 3, 3, 8])
print("x0:")
print(x0)
print()

r = 14

print("for lamb = 0:")
x_ = -np.linalg.inv(A) @ b
print("x_:", x_)
f_ = f(A, b, x_)
print("f(x_):", f_)
print("norm(x_ - x0):", np.linalg.norm(x_ - x0), "< r =", r)
print("for lamb > 0:")
lamb = 2
eps = 1e-17

#Итерационный процесс, для поиска решения
for i in range(8):
    print("i =", i)
    x_k = x0.copy()
    x_k1 = x0.copy()
    lamb_k = lamb
    x_k[i // 2] += ((-1) ** i) * r
    print("x_k start:")
    print(x_k)

    Jacobi = Jacobi_mat(A, x_k, x0, lamb_k)
    L_ = L(A, x_k, x0, lamb_k, b, r).astype(float)

    x_tmp = np.hstack((x_k, lamb_k)).astype(float)
    x_tmp -= np.linalg.inv(Jacobi) @ L_

    lamb_k = x_tmp[4]
    x_k1 = x_tmp[:4]

    while np.linalg.norm(x_k1 - x_k) > eps:
        x_k = x_k1
        Jacobi = Jacobi_mat(A, x_k, x0, lamb_k)
        L_ = L(A, x_k, x0, lamb_k, b, r).astype(float)

        x_tmp = np.hstack((x_k, lamb_k)).astype(float)
        x_tmp -= np.linalg.inv(Jacobi) @ L_

        lamb_k = x_tmp[4]
        x_k1 = x_tmp[:4]

    print("x_k end:")
    print(x_k1)
    print("lamb_k:", lamb_k)
    print("f(x):", f(A, b, x_k1))
    print("--------------")