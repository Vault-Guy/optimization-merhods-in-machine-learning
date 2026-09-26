from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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


def quad_grid(A, b=None, lim=2.0, n=320):
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y)
    P = np.stack([X, Y], axis=-1)
    Z = 0.5 * np.einsum("...i,ij,...j->...", P, A, P)
    if b is not None:
        Z -= P[..., 0] * b[0] + P[..., 1] * b[1]
    return X, Y, Z


# 1. Для общей функции Тейлор приближает, для квадратичной совпадает точно.
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
xx = np.linspace(-1.6, 1.6, 600)
x0 = 0.25
# общая функция
f = np.exp(0.6 * xx) + 0.08 * xx**3
f0 = np.exp(0.6 * x0) + 0.08 * x0**3
fp0 = 0.6*np.exp(0.6*x0) + 0.24*x0**2
fpp0 = 0.36*np.exp(0.6*x0) + 0.48*x0
qt = f0 + fp0*(xx-x0) + 0.5*fpp0*(xx-x0)**2
axes[0].plot(xx, f, lw=2.2, label="функция")
axes[0].plot(xx, qt, "--", lw=2.0, label="квадратичная модель")
axes[0].scatter([x0], [f0], s=35)
axes[0].set_title("Общая гладкая функция")
axes[0].legend(frameon=False)
axes[0].set_xlabel("x")
# квадратичная функция
q = 1.2*xx**2 - 0.8*xx + 0.4
q0 = 1.2*x0**2 - 0.8*x0 + 0.4
qp0 = 2.4*x0 - 0.8
qpp0 = 2.4
qt2 = q0 + qp0*(xx-x0) + 0.5*qpp0*(xx-x0)**2
axes[1].plot(xx, q, lw=2.2, label="функция")
axes[1].plot(xx, qt2, "--", lw=2.0, label="квадратичная модель")
axes[1].scatter([x0], [q0], s=35)
axes[1].set_title("Квадратичная функция")
axes[1].legend(frameon=False)
axes[1].set_xlabel("x")
save(fig, "quadratic_taylor_exact")

# 2. Три геометрии: круглая чаша, вытянутая чаша, седло.
fig = plt.figure(figsize=(9.2, 3.1))
for i, (A, title) in enumerate([
    (np.diag([1.0, 1.0]), "круглая чаша"),
    (np.diag([1.0, 10.0]), "вытянутая чаша"),
    (np.diag([1.0, -1.0]), "седло"),
], start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    x = np.linspace(-1.6, 1.6, 90)
    y = np.linspace(-1.6, 1.6, 90)
    X, Y = np.meshgrid(x, y)
    Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2)
    ax.plot_surface(X, Y, Z, alpha=0.72, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$f$")
    ax.view_init(elev=28, azim=-56)
save(fig, "three_quadratic_geometries")

# 3. Роль A, b, c.
A = np.array([[2.0, 0.5],[0.5, 1.0]])
fig = plt.figure(figsize=(9.1, 3.15))
x = np.linspace(-2.0, 2.0, 90)
y = np.linspace(-2.0, 2.0, 90)
X,Y = np.meshgrid(x,y)
P = np.stack([X,Y], axis=-1)
Z0 = 0.5*np.einsum('...i,ij,...j->...', P, A, P)
for i,(b,c,title) in enumerate([
    (np.array([0.0,0.0]),0.0,r"только $A$: форма"),
    (np.array([1.2,-0.8]),0.0,r"добавили $b$: сдвиг минимума"),
    (np.array([1.2,-0.8]),2.0,r"добавили $c$: вертикальный сдвиг"),
], start=1):
    ax = fig.add_subplot(1,3,i,projection='3d')
    Z = Z0 - b[0]*X - b[1]*Y + c
    ax.plot_surface(X,Y,Z,alpha=0.72,linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.view_init(elev=27, azim=-58)
save(fig,"quadratic_terms_effect")

# 4. Контуры конкретной квадратичной функции.
A = np.array([[2.0,1.0],[1.0,4.0]])
b = np.array([1.0,2.0])
X,Y,Z = quad_grid(A,b,lim=2.0)
fig, ax = plt.subplots(figsize=(5.2,3.7))
levels = np.linspace(Z.min()+0.15, min(Z.min()+8.0, Z.max()), 9)
cs = ax.contour(X,Y,Z,levels=levels,linewidths=1.2)
ax.clabel(cs, fontsize=8, fmt="%.1f")
xstar = np.linalg.solve(A,b)
ax.scatter([xstar[0]],[xstar[1]],s=55,zorder=5)
ax.text(xstar[0]+0.08,xstar[1]+0.06,r"$x^\star$")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$",rotation=0,labelpad=10)
ax.set_aspect('equal', adjustable='box')
ax.set_title(r"Линии уровня $\frac{1}{2}x^T A x-b^T x$")
save(fig,"quadratic_example_contours")

# 5. Кривизна общей и квадратичной функций в 1D.
fig, axes = plt.subplots(2,1,figsize=(6.0,4.3),sharex=True)
xx = np.linspace(-1.8,1.8,500)
f = 0.25*xx**4 + 0.5*xx**2
q = xx**2
axes[0].plot(xx,f,lw=2.1,label=r"$f(x)=x^4/4+x^2/2$")
axes[0].plot(xx,q,lw=2.1,label=r"$q(x)=x^2$")
axes[0].legend(frameon=False,loc="upper center",ncol=2)
axes[0].set_ylabel("значение")
axes[0].set_title("Форма функций")
axes[1].plot(xx,3*xx**2+1,lw=2.1,label=r"$f''(x)=3x^2+1$")
axes[1].plot(xx,np.full_like(xx,2.0),lw=2.1,label=r"$q''(x)=2$")
axes[1].legend(frameon=False,loc="upper center",ncol=2)
axes[1].set_ylabel("кривизна")
axes[1].set_xlabel("x")
axes[1].set_title("У квадратичной функции кривизна постоянна")
save(fig,"constant_curvature_quadratic")

# 6. Круглые и вытянутые линии уровня.
def contour_figure(A,name,title,lim=2.2):
    X,Y,Z=quad_grid(A,lim=lim)
    fig,ax=plt.subplots(figsize=(4.2,3.6))
    positive=Z-Z.min()
    levels=np.quantile(positive,[0.10,0.20,0.35,0.50,0.68])
    ax.contour(X,Y,Z,levels=levels,linewidths=1.25)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$",rotation=0,labelpad=8)
    ax.set_aspect('equal',adjustable='box'); ax.set_title(title)
    save(fig,name)
contour_figure(np.eye(2),"contours_equal_eigs",r"$\lambda_1=\lambda_2$: окружности")
contour_figure(np.diag([1.0,10.0]),"contours_unequal_eigs",r"$\lambda_1\ne\lambda_2$: эллипсы")

# 7. Сечения вдоль собственных направлений.
fig, ax = plt.subplots(figsize=(5.6,3.5))
t=np.linspace(-1.5,1.5,500)
ax.plot(t,0.5*1.0*t**2,lw=2.2,label=r"вдоль $v_1$, $\lambda_1=1$")
ax.plot(t,0.5*10.0*t**2,lw=2.2,label=r"вдоль $v_2$, $\lambda_2=10$")
ax.set_xlabel("t")
ax.set_ylabel(r"$f(tv_i)$")
ax.legend(frameon=False)
ax.set_title("Большое собственное значение = большая кривизна")
save(fig,"eigenvalue_curvature_sections")

# 8. Повернутые линии уровня и собственные направления.
theta=np.deg2rad(32)
Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
Lam=np.diag([1.0,6.0]); A=Q@Lam@Q.T
X,Y,Z=quad_grid(A,lim=2.4)
fig,ax=plt.subplots(figsize=(5.0,3.8))
levels=np.quantile(Z,[0.08,0.16,0.28,0.42,0.60])
ax.contour(X,Y,Z,levels=levels,linewidths=1.25)
for i,scale in [(0,1.7),(1,1.1)]:
    v=Q[:,i]
    ax.annotate("",xy=scale*v,xytext=-scale*v,arrowprops={"arrowstyle":"<->","lw":1.8})
    ax.text(*(scale*v+np.array([0.08,0.05])),rf"$v_{i+1}$")
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$",rotation=0,labelpad=8)
ax.set_aspect('equal',adjustable='box')
ax.set_title("Главные оси эллипса = собственные направления")
save(fig,"rotated_contours_eigenvectors")

# 9. Геометрия спектра: собственные значения видны и по полуосям, и по сечениям.
lam1, lam2 = 1.0, 6.0
a1, a2 = np.sqrt(2.0 / lam1), np.sqrt(2.0 / lam2)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55))

# Левый рисунок: линия уровня f=1 с точными полуосями вдоль собственных векторов.
ax = axes[0]
theta_grid = np.linspace(0, 2*np.pi, 500)
Zeig = np.vstack([a1*np.cos(theta_grid), a2*np.sin(theta_grid)])
ellipse = Q @ Zeig
ax.plot(ellipse[0], ellipse[1], lw=2.2)
for i, (lam, a) in enumerate([(lam1, a1), (lam2, a2)]):
    v = Q[:, i]
    ax.annotate("", xy=a*v, xytext=-a*v,
                arrowprops={"arrowstyle":"<->", "lw":2.0})
    # labels are placed away from the arrows to avoid overlap
    offset = np.array([0.08, 0.11]) if i == 0 else np.array([-0.72, 0.10])
    pos = a*v + offset
    ax.text(pos[0], pos[1],
            rf"$v_{i+1}$: $\lambda_{i+1}={lam:g}$" + "\n" + rf"$a_{i+1}=\sqrt{{2/\lambda_{i+1}}}={a:.2f}$",
            fontsize=9, bbox={"boxstyle":"round,pad=0.2", "facecolor":"white", "alpha":0.9, "edgecolor":"0.8"})
ax.scatter([0], [0], s=26)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$", rotation=0, labelpad=8)
ax.set_title(r"Линия уровня $f=1$")

# Правый рисунок: одномерные сечения вдоль собственных направлений.
ax = axes[1]
t = np.linspace(-1.45, 1.45, 500)
y1 = 0.5*lam1*t**2
y2 = 0.5*lam2*t**2
ax.plot(t, y1, lw=2.2, label=r"$v_1$: $f(tv_1)=\frac{1}{2}\cdot1\cdot t^2$")
ax.plot(t, y2, lw=2.2, label=r"$v_2$: $f(tv_2)=\frac{1}{2}\cdot6\cdot t^2$")
ax.scatter([1, 1], [0.5*lam1, 0.5*lam2], s=30)
ax.annotate(r"$f(v_1)=0.5$", xy=(1, 0.5), xytext=(0.25, 0.85),
            arrowprops={"arrowstyle":"->", "lw":1.1}, fontsize=9)
ax.annotate(r"$f(v_2)=3$", xy=(1, 3.0), xytext=(0.30, 3.35),
            arrowprops={"arrowstyle":"->", "lw":1.1}, fontsize=9)
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-0.05, 4.2)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$f(tv_i)$")
ax.set_title("Сечения вдоль главных направлений")
ax.legend(frameon=False, loc="upper center", fontsize=8)
save(fig, "spectrum_geometry")

# 10. Положительно определенная чаша.
fig=plt.figure(figsize=(5.0,3.7)); ax=fig.add_subplot(111,projection='3d')
x=np.linspace(-1.6,1.6,100); y=np.linspace(-1.6,1.6,100); X,Y=np.meshgrid(x,y)
Z=0.5*(X**2+3*Y**2)
ax.plot_surface(X,Y,Z,alpha=0.78,linewidth=0)
ax.scatter([0],[0],[0],s=45)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$f$")
ax.view_init(elev=28,azim=-58); ax.set_title(r"$A\succ0$: единственная чаша")
save(fig,"pd_bowl")

# 11. Седло.
fig=plt.figure(figsize=(5.0,3.7)); ax=fig.add_subplot(111,projection='3d')
Z=0.5*(X**2-Y**2)
ax.plot_surface(X,Y,Z,alpha=0.78,linewidth=0)
ax.scatter([0],[0],[0],s=45)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$f$")
ax.view_init(elev=28,azim=-58); ax.set_title("Собственные значения разных знаков: седло")
save(fig,"indefinite_saddle")

# 12. Полуопределенная долина.
fig=plt.figure(figsize=(5.0,3.7)); ax=fig.add_subplot(111,projection='3d')
Z=0.5*X**2
ax.plot_surface(X,Y,Z,alpha=0.78,linewidth=0)
ax.plot(np.zeros_like(y),y,np.zeros_like(y),lw=3.0)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$f$")
ax.view_init(elev=28,azim=-58); ax.set_title(r"$A\succeq0$: минимум вдоль целой прямой")
save(fig,"psd_valley")

# 13. Линейный член сдвигает центр, но не форму.
A=np.array([[3.0,0.8],[0.8,1.5]])
fig,axes=plt.subplots(1,2,figsize=(7.6,3.3))
for ax,b,title in [(axes[0],np.zeros(2),r"$b=0$"),(axes[1],np.array([1.8,-0.8]),r"$b\ne0$")]:
    X,Y,Z=quad_grid(A,b,lim=2.4)
    lev=np.quantile(Z-Z.min(),[0.08,0.17,0.30,0.46,0.65]) + Z.min()
    ax.contour(X,Y,Z,levels=lev,linewidths=1.25)
    xs=np.linalg.solve(A,b) if np.linalg.norm(b)>0 else np.zeros(2)
    ax.scatter([xs[0]],[xs[1]],s=45)
    ax.text(xs[0]+0.08,xs[1]+0.05,r"$x^\star$")
    ax.set_aspect('equal',adjustable='box'); ax.set_title(title)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$",rotation=0,labelpad=8)
save(fig,"linear_term_shifts_center")

# 14. Число обусловленности и точная вытянутость линии уровня.
fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.05))
for ax, kappa in zip(axes, [1, 10, 100]):
    a_long = np.sqrt(2.0)
    a_short = np.sqrt(2.0 / kappa)
    th = np.linspace(0, 2*np.pi, 500)
    xx = a_long*np.cos(th)
    yy = a_short*np.sin(th)
    ax.plot(xx, yy, lw=2.1)
    ax.axhline(0, lw=0.7, alpha=0.35)
    ax.axvline(0, lw=0.7, alpha=0.35)
    ax.annotate("", xy=(a_long, 0), xytext=(-a_long, 0),
                arrowprops={"arrowstyle":"<->", "lw":1.5})
    ax.annotate("", xy=(0, a_short), xytext=(0, -a_short),
                arrowprops={"arrowstyle":"<->", "lw":1.5})
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.62, 1.62)
    ax.set_title(rf"$\kappa={kappa}$,  $a_{{max}}/a_{{min}}=\sqrt{{\kappa}}={np.sqrt(kappa):.2g}$")
    ax.set_xlabel(r"$z_{\min}$")
    if ax is axes[0]:
        ax.set_ylabel(r"$z_{\max}$", rotation=0, labelpad=10)
fig.suptitle(r"Одна и та же линия уровня $\frac{1}{2} z^T\Lambda z=1$", y=1.02)
save(fig, "condition_number_contours")

# 15. Чувствительность решения к одному и тому же относительному возмущению b.
fig, ax = plt.subplots(figsize=(6.8, 3.4))
labels = [r"$\kappa=1$", r"$\kappa=100$"]
relative_dx = np.array([0.1, 10.0])
bars = ax.bar(labels, relative_dx)
ax.axhline(0.1, ls="--", lw=1.2, label=r"$\|\delta b\|/\|b\|=10\%$")
ax.set_ylabel(r"$\|\delta x\|/\|x^\star\|$")
ax.set_title("Одинаковое возмущение данных может по-разному сдвинуть решение")
ax.set_ylim(0, 11.3)
for bar, value, text in zip(bars, relative_dx, ["10%", "1000%"]):
    ax.text(bar.get_x()+bar.get_width()/2, value+0.25, text, ha="center", va="bottom")
ax.legend(frameon=False, loc="upper left")
save(fig, "conditioning_sensitivity")

# 16. Предпросмотр градиентного шага на круглой и вытянутой геометрии.
fig,axes=plt.subplots(1,2,figsize=(7.4,3.35))
for ax,A,title in [(axes[0],np.eye(2),"хорошая геометрия"),(axes[1],np.diag([1.0,20.0]),"плохая геометрия")]:
    X,Y,Z=quad_grid(A,lim=2.1)
    ax.contour(X,Y,Z,levels=[0.3,0.7,1.3,2.2,3.4],linewidths=1.1)
    p=np.array([1.4,1.0])
    g=A@p
    d=-g/np.linalg.norm(g)
    ax.scatter([p[0]],[p[1]],s=42)
    ax.scatter([0],[0],s=90,marker='*')
    ax.annotate("",xy=p+0.65*d,xytext=p,arrowprops={"arrowstyle":"->","lw":2.0})
    ax.text(p[0]+0.04,p[1]+0.08,r"$x$")
    ax.text(0.07,0.07,r"$x^\star$")
    ax.text(*(p+0.68*d+np.array([0.03,0.03])),r"$-\nabla f(x)$")
    ax.set_aspect('equal',adjustable='box'); ax.set_xlim(-2,2); ax.set_ylim(-2,2)
    ax.set_title(title); ax.set_xlabel(r"$x_1$")
    if ax is axes[0]: ax.set_ylabel(r"$x_2$",rotation=0,labelpad=8)
save(fig,"gradient_step_preview")

print(f"Generated figures in {OUT}")