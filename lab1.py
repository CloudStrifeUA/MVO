import numpy as np
import matplotlib.pyplot as plt

N = 201
Quiver_N = 21
size = 5
wolf_min = (-3/11, -18/11)  #Computed by WolframAlpha
colors = ('orange', 'green', 'red')

# Цільова функція
def func(x1, x2):
    return 3*x1*x1 + x2*x2 - x1*x2 + np.abs(x1 - x2 - 3) + np.abs(x1 + 2*x2 + 5)

def get_mesh(size, N):
    x1 = np.linspace(-size, size, N)
    x2 = np.linspace(-size, size, N)
    x1, x2 = np.meshgrid(x1, x2)
    f = func(x1, x2)

    l1x1 = np.linspace(-size, size, N)
    l1x2 = l1x1 - 3
    l1f = func(l1x1, l1x2)
    l2x1 = np.linspace(-size, size, N)
    l2x2 = (-5 - l2x1)/2
    l2f = func(l2x1, l2x2)
    return x1, x2, f, l1x1, l1x2, l1f, l2x1, l2x2, l2f

f_min = func(wolf_min[0], wolf_min[1])

#Субградієнт цільової функції
def f_subgrad(x1, x2):
    g = [
        6*x1 - x2,
        2*x2 - x1
    ]
    a = x1 - x2 - 3
    if a > 0:
        g[0] += 1
        g[1] += -1
    elif a == 0:
        g[0] += np.random.uniform(-1, 1, g[0].shape)
        g[1] += np.random.uniform(-1, 1, g[1].shape)
    else:
        g[0] += -1
        g[1] += 1

    b = x1 + 2*x2 + 5
    if b > 0:
        g[0] += 1
        g[1] += 2
    elif b == 0:
        g[0] += np.random.uniform(-1, 1, g[0].shape)
        g[1] += np.random.uniform(-2, 2, g[1].shape)
    else:
        g[0] += -1
        g[1] += -2

    return np.array(g)

# Субградієнтний метод
def optimize(func, sub_grad, init, step_func=lambda k: 1/(2 + k), max_iter=10000, tol=1e-6):

    points = [init]
    values = [func(init[0], init[1])]

    for k in range(max_iter):
        g = sub_grad(points[-1][0], points[-1][1])

        step_size = step_func(k)

        points.append(points[-1] - step_size * g)
        values.append(func(points[-1][0], points[-1][1]))

        if np.linalg.norm(points[-1] - points[-2]) < tol and \
           np.linalg.norm(values[-1] - values[-2]) < tol:
            break
    return points[-1], values[-1]


point, value = optimize(func, f_subgrad, np.array([0., 0.]))

print('Точка мінімуму порахована за допомогую субградієнтного методу:')
print(point)
print('Значення функції в цій точці:')
print(value)
print('Точка мінімуму порахована за допомогую WolframAlpha:')
print(wolf_min)
print('Значення функції в цій точці:')
print(f_min)

x1, x2, f, l1x1, l1x2, l1f, l2x1, l2x2, l2f = get_mesh(size, N)

#Малюнки
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlim((-size, size))
ax.set_ylim((-size, size))
ax.set_title("Тривимірний графік функції з виділеними негладкими частинами")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
surf = ax.plot_surface(x1, x2, f, cmap='cool', edgecolor ='none', alpha=0.7)
fig.colorbar(surf, ax = ax)
ax.plot3D(l1x1, l1x2, l1f, colors[0], label='nonsmooth part 1')
ax.plot3D(l2x1, l2x2, l2f, colors[1], label='nonsmooth part 2')
ax.scatter([point[0]], [point[1]], [f_min], c=colors[2], label='f min')
ax.legend(loc='best')

x1, x2, f, l1x1, l1x2, l1f, l2x1, l2x2, l2f = get_mesh(size, Quiver_N)
for _ in range(4):
    x1 = np.vstack([x1, l1x1])
    x2 = np.vstack([x2, l1x2])
    x1 = np.vstack([x1, l2x1])
    x2 = np.vstack([x2, l2x2])
qf = func(x1, x2)

g = np.array([f_subgrad(x1i, x2i) for (x1i, x2i) in zip(x1.ravel(), x2.ravel())]).reshape((*x1.shape, 2))
u = np.array([np.array([dx for (dx, dy) in gi]) for gi in g])
v = np.array([np.array([dy for (dx, dy) in gi]) for gi in g])
w = 0.5 * np.sqrt(u**2 + v**2)
u /= w
v /= w
ax = fig.add_subplot(1, 2, 2)
ax.quiver(x1, x2, -u, -v, label='gradient')
ax.plot(l1x1, l1x2, colors[0], label='x1 - x2 - 3 = 0')
ax.plot(l2x1, l2x2, colors[1], label='x1 + 2*x2 + 5 = 0')
ax.scatter([point[0]], [point[1]], c=colors[2], label='min')
ax.set_xlim((-size, size))
ax.set_ylim((-size, size))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title("Субдиференціал функції")
ax.legend(loc='best')
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()