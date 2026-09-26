from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Единый аккуратный стиль для рисунков лекции.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.autolayout": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# 1. Скалярное произведение и угол -------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.set_aspect("equal")
ax.set_xlim(-0.15, 3.6)
ax.set_ylim(-0.15, 2.7)
ax.axis("off")

u = np.array([3.0, 0.0])
v = np.array([2.15, 1.75])
ax.annotate("", xy=u, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.0})
ax.annotate("", xy=v, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.0})
ax.text(u[0] + 0.08, u[1] - 0.02, r"$u$", va="center")
ax.text(v[0] + 0.05, v[1] + 0.05, r"$v$")

theta_deg = np.degrees(np.arctan2(v[1], v[0]))
arc = Arc((0, 0), 1.55, 1.55, angle=0, theta1=0, theta2=theta_deg, lw=1.8)
ax.add_patch(arc)
mid = np.radians(theta_deg / 2)
ax.text(1.02 * np.cos(mid), 1.02 * np.sin(mid), r"$\theta$", ha="center", va="center", fontsize=14)
save(fig, "angle_dot_product")


# 2. Линейное преобразование A = diag(2, 1/2) ------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.7))
ax.set_aspect("equal")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-1.8, 1.8)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$", rotation=0, labelpad=10)

t = np.linspace(0, 2 * np.pi, 500)
circle = np.column_stack([np.cos(t), np.sin(t)])
A = np.array([[2.0, 0.0], [0.0, 0.5]])
ellipse = circle @ A.T
ax.plot(circle[:, 0], circle[:, 1], lw=2.0, label="единичная окружность")
ax.plot(ellipse[:, 0], ellipse[:, 1], lw=2.0, linestyle="--", label=r"образ $A(S^1)$")

# Показываем действие на базис и на одну произвольную точку.
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
for vec, label, dy in [(e1, r"$e_1$", -0.16), (e2, r"$e_2$", 0.06)]:
    Avec = A @ vec
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.annotate("", xy=Avec, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 1.8, "linestyle": "--"})
    ax.text(vec[0] + 0.05, vec[1] + dy, label)

x = np.array([0.75, 0.75])
Ax = A @ x
ax.scatter(*x, s=30)
ax.scatter(*Ax, s=30)
ax.annotate("", xy=Ax, xytext=x, arrowprops={"arrowstyle": "->", "lw": 1.4, "linestyle": ":"})
ax.text(*(x + np.array([0.05, 0.08])), r"$x$")
ax.text(*(Ax + np.array([0.05, -0.18])), r"$Ax$")
ax.text(-2.45, -1.55, r"$(x_1,x_2)\mapsto(2x_1,\,x_2/2)$", fontsize=10)
ax.legend(loc="upper right", frameon=False)
save(fig, "matrix_transform")


# 3. Ортогональные векторы --------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.set_aspect("equal")
ax.axhline(0, linewidth=0.7)
ax.axvline(0, linewidth=0.7)
ax.set_xlim(-0.6, 2.8)
ax.set_ylim(-1.6, 2.8)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$", rotation=0, labelpad=10)

u = np.array([1.0, 2.0])
v = np.array([2.0, -1.0])
ax.annotate("", xy=u, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.0})
ax.annotate("", xy=v, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.0})
ax.text(u[0] + 0.05, u[1] + 0.08, r"$u=(1,2)$")
ax.text(v[0] + 0.05, v[1] - 0.05, r"$v=(2,-1)$", va="top")

uhat = u / np.linalg.norm(u)
vhat = v / np.linalg.norm(v)
s = 0.38
p0 = np.array([0.0, 0.0])
p1 = s * uhat
p2 = s * (uhat + vhat)
p3 = s * vhat
ax.add_patch(Polygon([p0, p1, p2, p3], closed=False, fill=False, lw=1.2))
ax.text(0.65, 0.15, r"$90^\circ$")
save(fig, "orthogonal_vectors")


# 4. Кривизна и аппроксимации Тейлора --------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.6))
t = np.linspace(-1.1, 1.1, 500)
f = np.exp(t)
linear = 1 + t
quadratic = 1 + t + 0.5 * t**2
ax.plot(t, f, lw=2.3, label=r"$f(t)=e^t$")
ax.plot(t, linear, lw=1.8, linestyle="--", label="линейная модель")
ax.plot(t, quadratic, lw=1.8, linestyle=":", label="квадратичная модель")
ax.scatter([0], [1], s=35, zorder=5)
ax.axvline(0, linewidth=0.7, linestyle="--")
ax.set_xlabel(r"смещение $h$")
ax.set_ylabel("значение")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Второй порядок учитывает изменение наклона")
save(fig, "curvature_taylor_1d")


# 5. Поверхность и линии уровня ---------------------------------------------
fig = plt.figure(figsize=(5.8, 4.25))
ax = fig.add_subplot(111, projection="3d")
x = np.linspace(-2.0, 2.0, 120)
y = np.linspace(-1.6, 1.6, 120)
X, Y = np.meshgrid(x, y)
Z = X**2 + 2 * Y**2
ax.plot_surface(X, Y, Z, alpha=0.38, linewidth=0, antialiased=True)
ax.contour(X, Y, Z, levels=[1, 2, 4, 6, 8], zdir="z", offset=-0.8, linewidths=1.2)
ax.set_zlim(-0.8, 9.0)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$f$")
ax.view_init(elev=28, azim=-58)
save(fig, "surface_and_contours")


# 6. Градиент не обязан смотреть на глобальный минимум ---------------------
# Чистая иллюстрация на функции Розенброка: локальный антиградиент и прямая
# к минимуму заметно различаются по направлению.
fig, ax = plt.subplots(figsize=(5.7, 4.0))
x = np.linspace(-0.9, 1.35, 500)
y = np.linspace(-0.25, 1.65, 500)
X, Y = np.meshgrid(x, y)
Z = 10 * (Y - X**2) ** 2 + (X - 1) ** 2
levels = [0.1, 0.3, 0.7, 1.5, 3, 6, 12, 24]
cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.0)
ax.clabel(cs, fmt="%g", fontsize=7)

p = np.array([0.0, 1.0])
minimum = np.array([1.0, 1.0])
grad = np.array([-2.0, 20.0])
desc = -grad / np.linalg.norm(grad)
direct = (minimum - p) / np.linalg.norm(minimum - p)

ax.scatter([p[0]], [p[1]], s=55, zorder=6)
ax.scatter([minimum[0]], [minimum[1]], s=100, marker="*", zorder=6)
ax.annotate("", xy=p + 0.52 * desc, xytext=p,
            arrowprops={"arrowstyle": "->", "lw": 2.4})
ax.annotate("", xy=p + 0.72 * direct, xytext=p,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "linestyle": "--"})
ax.text(p[0] - 0.08, p[1] + 0.10, r"$x$")
ax.text(minimum[0] + 0.04, minimum[1] + 0.04, r"$x^\star$")
ax.text(0.08, 0.50, r"$-\nabla f(x)$", fontsize=11)
ax.text(0.36, 1.08, "прямая к минимуму", fontsize=9)
ax.text(-0.73, 1.48, "изогнутая долина", fontsize=9)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$", rotation=0, labelpad=10)
ax.set_xlim(-0.85, 1.3)
ax.set_ylim(-0.2, 1.6)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Локально лучший шаг не обязан вести прямо к минимуму")
save(fig, "gradient_not_to_minimum")


# 7. Линии уровня, градиент и касательное направление ----------------------
fig, ax = plt.subplots(figsize=(5.2, 3.8))
x = np.linspace(-2.2, 2.2, 350)
y = np.linspace(-1.7, 1.7, 350)
X, Y = np.meshgrid(x, y)
Z = X**2 + 2 * Y**2
cs = ax.contour(X, Y, Z, levels=[0.5, 1, 2, 3, 5, 7], linewidths=1.0)
ax.clabel(cs, inline=True, fontsize=8)

p = np.array([1.0, 0.7])
g = np.array([2.0 * p[0], 4.0 * p[1]])
gh = g / np.linalg.norm(g)
tangent = np.array([g[1], -g[0]])
tangent = tangent / np.linalg.norm(tangent)
ax.scatter([p[0]], [p[1]], s=42, zorder=5)
ax.annotate("", xy=p + 0.75 * gh, xytext=p,
            arrowprops={"arrowstyle": "->", "lw": 2.0})
ax.annotate("", xy=p + 0.75 * tangent, xytext=p,
            arrowprops={"arrowstyle": "->", "lw": 1.7, "linestyle": "--"})
ax.text(*(p + 0.78 * gh + np.array([0.03, 0.03])), r"$\nabla f(x)$")
ax.text(*(p + 0.76 * tangent + np.array([-0.15, 0.05])), "касательная")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$", rotation=0, labelpad=10)
ax.set_aspect("equal", adjustable="box")
save(fig, "level_sets_gradient")


# 8. Круглые и вытянутые линии уровня --------------------------------------
for name, factor, title in [
    ("contours_round", 1.0, r"$x_1^2+x_2^2$"),
    ("contours_elongated", 10.0, r"$x_1^2+10x_2^2$"),
]:
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    x = np.linspace(-2.2, 2.2, 350)
    y = np.linspace(-2.2, 2.2, 350)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + factor * Y**2
    ax.contour(X, Y, Z, levels=[0.5, 1, 2, 3, 4], linewidths=1.1)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0, labelpad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    save(fig, name)


# 9. Кривизна вдоль направления как кривизна одномерного сечения -----------
fig = plt.figure(figsize=(5.7, 4.2))
ax = fig.add_subplot(111, projection="3d")
x = np.linspace(-1.6, 1.6, 110)
y = np.linspace(-1.4, 1.4, 110)
X, Y = np.meshgrid(x, y)
Z = 0.5 * X**2 + 2.0 * Y**2 + 0.3 * X * Y
ax.plot_surface(X, Y, Z, alpha=0.30, linewidth=0)

x0 = np.array([0.35, -0.25])
v = np.array([1.0, 0.65])
v = v / np.linalg.norm(v)
t = np.linspace(-1.1, 1.1, 180)
xs = x0[0] + t * v[0]
ys = x0[1] + t * v[1]
zs = 0.5 * xs**2 + 2.0 * ys**2 + 0.3 * xs * ys
ax.plot(xs, ys, zs, lw=3.0, label=r"$\varphi(t)=f(x+tv)$")
z0 = 0.5 * x0[0]**2 + 2.0 * x0[1]**2 + 0.3 * x0[0] * x0[1]
ax.scatter([x0[0]], [x0[1]], [z0], s=45)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$f$")
ax.view_init(elev=27, azim=-57)
ax.legend(frameon=False, loc="upper left")
save(fig, "directional_curvature")


# 10. Три типа локальной кривизны -------------------------------------------
def surface_icon(name: str, kind: str) -> None:
    fig = plt.figure(figsize=(2.8, 2.5))
    ax = fig.add_subplot(111, projection="3d")
    x = np.linspace(-1.2, 1.2, 45)
    y = np.linspace(-1.2, 1.2, 45)
    X, Y = np.meshgrid(x, y)
    if kind == "bowl":
        Z = X**2 + Y**2
    elif kind == "dome":
        Z = -(X**2 + Y**2)
    else:
        Z = X**2 - Y**2
    ax.plot_surface(X, Y, Z, alpha=0.48, linewidth=0)
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=0.45)
    ax.set_axis_off()
    ax.view_init(elev=27, azim=-55)
    save(fig, name)


surface_icon("curvature_bowl", "bowl")
surface_icon("curvature_dome", "dome")
surface_icon("curvature_saddle", "saddle")


# 11. Слайд 2: текущая точка, минимум и выбор направления ------------------
fig, ax = plt.subplots(figsize=(6.0, 2.5))
x = np.linspace(-2.0, 2.2, 300)
y = np.linspace(-1.3, 1.5, 250)
X, Y = np.meshgrid(x, y)
Z = 0.7 * (X - 1.0) ** 2 + 1.5 * (Y - 0.15) ** 2
ax.contour(X, Y, Z, levels=[0.3, 0.8, 1.5, 2.5, 4.0], linewidths=1.0)
current = np.array([-1.0, 0.65])
optimum = np.array([1.0, 0.15])
ax.scatter(*current, s=55, zorder=5)
ax.scatter(*optimum, s=100, marker="*", zorder=5)
for d in [np.array([0.72, 0.12]), np.array([0.50, -0.52]), np.array([0.25, 0.60])]:
    ax.annotate("", xy=current + d, xytext=current,
                arrowprops={"arrowstyle": "->", "lw": 1.5})
ax.text(current[0] - 0.12, current[1] + 0.12, r"$x$")
ax.text(optimum[0] + 0.08, optimum[1] + 0.04, r"$x^\star$")
ax.set_title("Какое направление выбрать?", pad=10, fontsize=11)
ax.set_xlabel("пространство параметров")
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")
save(fig, "optimization_step_direction")


# 12. Слайд 5: векторное обновление x -> x+h -------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.5))
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlim(-0.4, 3.3)
ax.set_ylim(-0.3, 3.6)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$", rotation=0, labelpad=10)
x0 = np.array([2.0, 1.0])
h = np.array([-1.0, 2.0])
x1 = x0 + h
ax.scatter(*x0, s=55, zorder=5)
ax.scatter(*x1, s=55, zorder=5)
ax.annotate("", xy=x1, xytext=x0,
            arrowprops={"arrowstyle": "->", "lw": 2.2})
ax.text(x0[0] + 0.08, x0[1] - 0.20, r"$x=(2,1)$")
ax.text(x1[0] + 0.08, x1[1] + 0.08, r"$x+h=(1,3)$")
ax.text(1.72, 2.10, r"$h=(-1,2)$", fontsize=11, ha="left", va="center")
ax.set_aspect("equal", adjustable="box")
save(fig, "vector_update")


# 13. Слайд 12: функция, касательная и линейная модель ----------------------
fig, ax = plt.subplots(figsize=(5.0, 3.5))
t = np.linspace(-2.0, 2.2, 500)
f = 0.22 * (t + 0.35) ** 3 + 0.20 * (t + 0.35) ** 2 + 0.7
x0 = 0.45
# Производная аналитически.
def fun(z):
    return 0.22 * (z + 0.35) ** 3 + 0.20 * (z + 0.35) ** 2 + 0.7

def der(z):
    return 0.66 * (z + 0.35) ** 2 + 0.40 * (z + 0.35)

y0 = fun(x0)
slope = der(x0)
tangent = y0 + slope * (t - x0)
ax.plot(t, f, lw=2.2, label=r"$f$")
ax.plot(t, tangent, lw=1.8, linestyle="--", label="линейная модель")
ax.scatter([x0], [y0], s=45, zorder=5)
ax.axvline(x0, linewidth=0.7, linestyle=":")
ax.text(x0 + 0.08, y0 - 0.38, r"$(x,f(x))$", ha="left", va="top")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$f(t)$", rotation=0, labelpad=10)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Около точки функция почти совпадает с касательной")
save(fig, "local_linear_model")


# 14. Слайд 24: градиент как поле направлений -------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.5))
x = np.linspace(-1.8, 1.8, 300)
y = np.linspace(-1.45, 1.75, 300)
X, Y = np.meshgrid(x, y)
Z = X**2 + 2 * Y**2
ax.contour(X, Y, Z, levels=[0.5, 1, 2, 3, 4.5, 6, 8], linewidths=0.9)
points = [np.array([1.0, 1.0]), np.array([-1.0, 1.0])]
for p0 in points:
    g = np.array([2 * p0[0], 4 * p0[1]], dtype=float)
    gh = g / np.linalg.norm(g)
    ax.scatter(*p0, s=55, zorder=6)
    ax.annotate("", xy=p0 + 0.62 * gh, xytext=p0,
                arrowprops={"arrowstyle": "-|>", "lw": 2.4, "mutation_scale": 14})
ax.text(0.72, 0.66, r"$\nabla f(1,1)$", fontsize=10, ha="left", va="top")
ax.text(-1.58, 0.66, r"$\nabla f(-1,1)$", fontsize=10, ha="left", va="top")
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.4, 1.72)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$", rotation=0, labelpad=8)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Одна функция, разные точки — разные градиенты")
save(fig, "gradient_field_two_points")


# 15. Собственный вектор: матрица сохраняет прямую направления -------------
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.set_xlim(-0.4, 3.4)
ax.set_ylim(-0.75, 2.4)
ax.axis("off")
# Общая прямая собственного направления.
ax.plot([-0.15, 3.10], [-0.10, 2.05], linewidth=1.0, linestyle=":")
origin = np.array([0.0, 0.0])
v = np.array([1.05, 0.70])
Av = np.array([2.55, 1.70])
ax.annotate("", xy=v, xytext=origin,
            arrowprops={"arrowstyle": "-|>", "lw": 2.4, "mutation_scale": 14})
ax.annotate("", xy=Av, xytext=origin,
            arrowprops={"arrowstyle": "-|>", "lw": 2.6, "mutation_scale": 14})
ax.scatter([0], [0], s=28, zorder=5)
ax.text(v[0] - 0.18, v[1] + 0.14, r"$v$", fontsize=13)
ax.text(Av[0] - 0.20, Av[1] + 0.13, r"$Av=\lambda v$", fontsize=13)
ax.text(0.12, -0.48, "оба вектора лежат на одной прямой", fontsize=10)
ax.text(0.38, 2.14, "матрица меняет масштаб, но не собственное направление", fontsize=9.5)
save(fig, "eigenvector_scaling")




# 16. Слайд 7: знак скалярного произведения в трёхмерном пространстве ------
fig = plt.figure(figsize=(5.8, 4.2))
ax = fig.add_subplot(111, projection="3d")

origin = np.zeros(3)
x_vec = np.array([1.15, 1.00, 0.85])
v_pos = np.array([1.05, 0.35, 0.55])
v_zero = np.array([1.00, -1.00, 0.00])
v_neg = np.array([-0.95, -0.55, -0.70])

# Немного нормируем вспомогательные векторы, чтобы длины были сопоставимы.
def scaled(v, length=1.35):
    return length * v / np.linalg.norm(v)

v_pos = scaled(v_pos)
v_zero = scaled(v_zero)
v_neg = scaled(v_neg)

vectors = [
    (x_vec, r"$x$", "black", 3.0),
    (v_pos, r"$v_+$", "C0", 2.2),
    (v_zero, r"$v_0$", "C2", 2.2),
    (v_neg, r"$v_-$", "C3", 2.2),
]

for vec, label, color, lw in vectors:
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
              arrow_length_ratio=0.10, linewidth=lw, color=color)
    ax.text(*(vec * 1.08), label, fontsize=12, color=color)

# Полупрозрачный прямоугольник показывает плоскость, ортогональную x.
# Берём два взаимно независимых направления, ортогональных x.
a = np.array([1.0, -1.0, 0.0])
a = a - x_vec * (a @ x_vec) / (x_vec @ x_vec)
a = a / np.linalg.norm(a)
b = np.cross(x_vec, a)
b = b / np.linalg.norm(b)
ss = np.linspace(-1.15, 1.15, 2)
tt = np.linspace(-1.15, 1.15, 2)
S, T = np.meshgrid(ss, tt)
PX = S * a[0] + T * b[0]
PY = S * a[1] + T * b[1]
PZ = S * a[2] + T * b[2]
ax.plot_surface(PX, PY, PZ, alpha=0.08, color="gray", shade=False)

ax.text2D(0.03, 0.94, r"$x^\top v_+>0$", transform=ax.transAxes, color="C0", fontsize=10)
ax.text2D(0.03, 0.87, r"$x^\top v_0=0$", transform=ax.transAxes, color="C2", fontsize=10)
ax.text2D(0.03, 0.80, r"$x^\top v_-<0$", transform=ax.transAxes, color="C3", fontsize=10)

lim = 1.65
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-1.35, 1.65)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
ax.view_init(elev=24, azim=-52)
ax.set_box_aspect((1, 1, 0.9))
save(fig, "dot_product_3d_signs")

print(f"Generated figures in {OUT}")