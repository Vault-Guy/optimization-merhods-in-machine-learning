import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
BLUE = '#003D7C'
RED = '#E52E5A'
GRAY = '#6B7280'
GREEN = '#2A7F62'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'mathtext.fontset': 'dejavusans',
    'font.family': 'DejaVu Sans',
})

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.06)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.06, dpi=180)
    plt.close(fig)

# 1. Convex vs nonconvex landscape
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
x = np.linspace(-3.2, 3.2, 700)
y1 = 0.22*x**2 + 0.18
axs[0].plot(x, y1, lw=2.3, color=BLUE)
axs[0].scatter([0], [0.18], s=55, color=RED, zorder=5)
axs[0].annotate('единственный\nглобальный минимум', xy=(0,0.18), xytext=(1.0,1.55),
                arrowprops=dict(arrowstyle='->', lw=1.1), fontsize=10)
axs[0].set_title('Выпуклая функция')
axs[0].set_xlabel('$x$'); axs[0].set_ylabel('$f(x)$'); axs[0].grid(alpha=.16)

y2 = 0.07*x**4 - 0.55*x**2 + 0.11*x + 1.1
axs[1].plot(x, y2, lw=2.3, color=BLUE)
idx = np.where((y2[1:-1] < y2[:-2]) & (y2[1:-1] < y2[2:]))[0]+1
for j,i in enumerate(idx[:2]):
    axs[1].scatter([x[i]], [y2[i]], s=50, color=RED if j==0 else GREEN, zorder=5)
axs[1].set_title('Невыпуклая функция')
axs[1].set_xlabel('$x$'); axs[1].grid(alpha=.16)
fig.suptitle('Убывание функции само по себе не говорит, в какой минимум мы придём', y=1.03)
fig.tight_layout(); save(fig, 'convex_vs_nonconvex')

# 2. Convex/nonconvex sets
fig, axs = plt.subplots(1, 4, figsize=(12.2, 3.0))
ax=axs[0]; ax.add_patch(Circle((0,0),1.0,facecolor=BLUE,alpha=.13,edgecolor=BLUE,lw=2)); ax.plot([-0.7,0.6],[-0.4,0.55],lw=2,color=RED); ax.scatter([-0.7,0.6],[-0.4,0.55],s=30,color=RED); ax.set_title('Шар — выпуклый')
ax=axs[1]; ax.add_patch(Rectangle((-1,-.7),2,1.4,facecolor=BLUE,alpha=.13,edgecolor=BLUE,lw=2)); ax.plot([-.8,.7],[.45,-.45],lw=2,color=RED); ax.scatter([-.8,.7],[.45,-.45],s=30,color=RED); ax.set_title('Прямоугольник — выпуклый')
ax=axs[2]; xx=np.linspace(-1.3,1.3,100); yy=0.35-0.5*xx; ax.fill_between(xx,-1.2,yy,color=BLUE,alpha=.13); ax.plot(xx,yy,lw=2,color=BLUE); ax.plot([-.8,.7],[-.35,-.2],lw=2,color=RED); ax.scatter([-.8,.7],[-.35,-.2],s=30,color=RED); ax.set_title('Полуплоскость — выпуклая')
ax=axs[3]; th=np.linspace(0,2*np.pi,400); outer=np.c_[1.0*np.cos(th),1.0*np.sin(th)]; inner=np.c_[0.62*np.cos(th)+0.35,0.62*np.sin(th)]; poly=np.vstack([outer, inner[::-1]])
ax.add_patch(Polygon(poly, closed=True, facecolor=BLUE, alpha=.13, edgecolor=BLUE, lw=2)); p1=(-.72,.42); p2=(.55,-.62); ax.plot([p1[0],p2[0]],[p1[1],p2[1]],lw=2,color=RED); ax.scatter([p1[0],p2[0]],[p1[1],p2[1]],s=30,color=RED); ax.set_title('Невыпуклое множество')
for ax in axs:
    ax.set_aspect('equal'); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.1,1.1); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
fig.tight_layout(); save(fig,'convex_sets')

# 3. Convex combination in triangle
fig, ax = plt.subplots(figsize=(6.8,4.0))
pts=np.array([[-1.7,-.7],[1.65,-.6],[.2,1.55]])
tri=Polygon(pts,closed=True,facecolor=BLUE,alpha=.10,edgecolor=BLUE,lw=2)
ax.add_patch(tri)
a=np.array([0.2,0.3,0.5]); z=(a[:,None]*pts).sum(axis=0)
ax.scatter(pts[:,0],pts[:,1],s=45,color=BLUE)
for i,p in enumerate(pts,1): ax.text(p[0]+.08,p[1]+.08,rf'$x_{i}$',fontsize=12)
ax.scatter([z[0]],[z[1]],s=60,color=RED,zorder=5); ax.text(z[0]+.08,z[1]+.08,r'$\sum_i \alpha_i x_i$',fontsize=12)
ax.text(-1.45,1.25,r'$\alpha_i\geq0,\quad \sum_i\alpha_i=1$',fontsize=12)
ax.set_xlim(-2.1,2.1); ax.set_ylim(-1.15,1.9); ax.set_aspect('equal'); ax.set_axis_off(); ax.set_title('Выпуклая комбинация остаётся внутри выпуклого множества')
fig.tight_layout(); save(fig,'convex_combination')

# 4. Chord condition
fig, axs = plt.subplots(1,2,figsize=(10.5,3.8))
x=np.linspace(-2.5,2.5,500); f=x**2/2+0.25
for ax, title, shift in [(axs[0],'Выпуклая функция',0),(axs[1],'Невыпуклый пример',1)]:
    if shift==0:
        yy=f
        xa,xb=-1.7,1.35
    else:
        yy=0.12*x**4-0.72*x**2+1.2
        xa,xb=-1.6,1.55
    ax.plot(x,yy,lw=2.2,color=BLUE)
    ya=np.interp(xa,x,yy); yb=np.interp(xb,x,yy)
    ax.plot([xa,xb],[ya,yb],lw=2,color=RED,ls='--')
    ax.scatter([xa,xb],[ya,yb],s=35,color=RED,zorder=5)
    xm=.5*(xa+xb); ym=np.interp(xm,x,yy); ych=.5*(ya+yb)
    ax.scatter([xm],[ym],s=35,color=BLUE,zorder=6)
    ax.vlines(xm,ym,ych,colors=GRAY,linestyles=':',lw=1.4)
    ax.set_title(title); ax.set_xlabel('$x$'); ax.grid(alpha=.15)
axs[0].set_ylabel('$f(x)$')
fig.suptitle('Хордовое определение: график выпуклой функции лежит не выше хорды',y=1.03)
fig.tight_layout(); save(fig,'chord_condition')

# 5. x^2 chord algebra illustration
fig, ax = plt.subplots(figsize=(7.4,4.0))
x=np.linspace(-2.4,2.4,500); y=x**2
xa,xb=-1.5,1.3; t=.35; z=(1-t)*xa+t*xb
ya=xa**2; yb=xb**2; yz=z**2; chord=(1-t)*ya+t*yb
ax.plot(x,y,lw=2.2,color=BLUE)
ax.plot([xa,xb],[ya,yb],ls='--',lw=2,color=RED)
ax.scatter([z],[yz],s=50,color=BLUE,zorder=5)
ax.scatter([z],[chord],s=50,color=RED,zorder=5)
ax.vlines(z,yz,chord,colors=GRAY,linestyles=':',lw=1.4)
ax.annotate(r'$f((1-t)x+ty)$',xy=(z,yz),xytext=(0.25,0.8),textcoords='axes fraction',arrowprops=dict(arrowstyle='->'))
ax.annotate(r'$(1-t)f(x)+tf(y)$',xy=(z,chord),xytext=(0.55,0.72),textcoords='axes fraction',arrowprops=dict(arrowstyle='->'))
ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$'); ax.grid(alpha=.15); ax.set_title(r'Для $f(x)=x^2$ точка графика лежит ниже хорды')
fig.tight_layout(); save(fig,'x2_chord')

# 6. Tangent is a global lower bound
fig, ax = plt.subplots(figsize=(7.8,4.2))
x=np.linspace(-2.5,2.6,600); y=np.exp(0.65*x)
x0=-.55; y0=np.exp(.65*x0); g=.65*y0; tangent=y0+g*(x-x0)
ax.plot(x,y,lw=2.3,color=BLUE,label='$f(x)$')
ax.plot(x,tangent,lw=2,color=RED,ls='--',label='касательная')
ax.scatter([x0],[y0],s=50,color=RED,zorder=5)
ax.fill_between(x,tangent,y,where=(y>=tangent),alpha=.08,color=BLUE)
ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$'); ax.set_title('У выпуклой функции касательная — глобальная нижняя оценка')
ax.legend(fontsize=9); ax.grid(alpha=.15); fig.tight_layout(); save(fig,'tangent_lower_bound')

# 7. Local min is global in convex case
fig, axs = plt.subplots(1,2,figsize=(10.5,3.8))
x=np.linspace(-3,3,700)
f=.15*(x-0.4)**2+.5
axs[0].plot(x,f,lw=2.2,color=BLUE); axs[0].scatter([.4],[.5],s=50,color=RED); axs[0].set_title('Выпуклая: локальный минимум глобален'); axs[0].grid(alpha=.15)
f2=.06*x**4-.42*x**2+.08*x+1.0
axs[1].plot(x,f2,lw=2.2,color=BLUE); idx=np.where((f2[1:-1]<f2[:-2])&(f2[1:-1]<f2[2:]))[0]+1
for j,i in enumerate(idx): axs[1].scatter([x[i]],[f2[i]],s=45,color=RED if j==0 else GREEN,zorder=5)
axs[1].set_title('Невыпуклая: локальные минимумы могут отличаться'); axs[1].grid(alpha=.15)
for ax in axs: ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
fig.tight_layout(); save(fig,'local_global_minima')

# 8. Hessian curvature: convex bowl, flat direction, saddle
fig = plt.figure(figsize=(11.4,3.4))
for i,(lam1,lam2,title) in enumerate([(1,3,'$H\\succ0$'),(1,0,'$H\\succeq0$, есть плоское направление'),(1,-1,'Неопределённый $H$')],1):
    ax=fig.add_subplot(1,3,i,projection='3d')
    x=np.linspace(-1.8,1.8,80); y=np.linspace(-1.8,1.8,80); X,Y=np.meshgrid(x,y); Z=.5*(lam1*X**2+lam2*Y**2)
    ax.plot_surface(X,Y,Z,cmap='Blues',alpha=.85,linewidth=0,antialiased=True)
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(28,-55)
fig.suptitle('Знак направленной кривизны определяется гессианом',y=1.02)
fig.tight_layout(); save(fig,'hessian_geometries')

# 9. Strict but not strong: x^4 vs strong x^2
fig, axs = plt.subplots(1,2,figsize=(10.3,3.8))
x=np.linspace(-1.6,1.6,500)
axs[0].plot(x,x**4,lw=2.3,color=BLUE); axs[0].set_title(r'$x^4$: строго выпукла, но очень плоская у 0')
axs[1].plot(x,x**2,lw=2.3,color=BLUE); axs[1].set_title(r'$x^2$: равномерная положительная кривизна')
for ax in axs: ax.axhline(0,lw=.6,alpha=.4); ax.axvline(0,lw=.6,alpha=.4); ax.grid(alpha=.15); ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
fig.tight_layout(); save(fig,'strict_vs_strong')

# 10. Strong convexity: tangent + parabola lower support
fig, ax = plt.subplots(figsize=(7.8,4.3))
x=np.linspace(-2.3,2.5,600)
f=.5*x**2+.07*x**4; x0=-.7; f0=.5*x0**2+.07*x0**4; g=x0+.28*x0**3; mu=1.0
lin=f0+g*(x-x0); lower=lin+.5*mu*(x-x0)**2
ax.plot(x,f,lw=2.3,color=BLUE,label='$f(y)$')
ax.plot(x,lin,lw=1.6,ls='--',color=GRAY,label='касательная')
ax.plot(x,lower,lw=2,color=RED,label=r'касательная $+\,\frac{\mu}{2}(y-x)^2$')
ax.scatter([x0],[f0],s=45,color=RED,zorder=5)
ax.fill_between(x,lower,f,where=f>=lower,alpha=.06,color=BLUE)
ax.set_title(r'Сильная выпуклость: функция лежит выше параболической опоры')
ax.set_xlabel('$y$'); ax.set_ylabel('значение'); ax.legend(fontsize=8.5); ax.grid(alpha=.15)
fig.tight_layout(); save(fig,'strong_convexity_support')

# 11. Ridge improves smallest eigenvalue / conditioning
fig, axs = plt.subplots(1,2,figsize=(10.8,3.8))
A=np.diag([0.12,2.5]); lam=.65
for ax,M,title in [(axs[0],A,'Без регуляризации'),(axs[1],A+lam*np.eye(2),r'Ridge: $A+\lambda I$')]:
    xx=np.linspace(-4,4,400); yy=np.linspace(-4,4,400); X,Y=np.meshgrid(xx,yy); Z=.5*(M[0,0]*X**2+M[1,1]*Y**2)
    ax.contour(X,Y,Z,levels=[.4,1.0,2.0,3.5],linewidths=1.5)
    eig=np.linalg.eigvalsh(M); ax.set_title(title+'\n'+rf'$\lambda_{{min}}={eig.min():.2f},\ \kappa={eig.max()/eig.min():.1f}$',fontsize=10.5)
    ax.set_aspect('equal'); ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Ridge добавляет кривизну во всех направлениях',y=1.02); fig.tight_layout(); save(fig,'ridge_geometry')

# 12. Logistic loss convexity
fig, axs = plt.subplots(1,2,figsize=(10.3,3.7))
z=np.linspace(-5,5,600); loss=np.log1p(np.exp(-z)); second=np.exp(z)/(1+np.exp(z))**2
axs[0].plot(z,loss,lw=2.3,color=BLUE); axs[0].set_title(r'$\ell(z)=\log(1+e^{-z})$'); axs[0].set_xlabel('$z$'); axs[0].set_ylabel('$\ell(z)$'); axs[0].grid(alpha=.15)
axs[1].plot(z,second,lw=2.3,color=RED); axs[1].axhline(0,lw=.8,color=GRAY); axs[1].set_title(r"$\ell''(z)>0$ для всех $z$"); axs[1].set_xlabel('$z$'); axs[1].set_ylabel(r"$\ell''(z)$"); axs[1].grid(alpha=.15)
fig.tight_layout(); save(fig,'logistic_convexity')

# 13. Nonconvex neural-network-like toy landscape
fig, ax=plt.subplots(figsize=(7.8,4.0))
x=np.linspace(-3.1,3.1,700); y=.035*x**6-.27*x**4+.48*x**2+.12*np.sin(4*x)+.5
ax.plot(x,y,lw=2.2,color=BLUE)
idx=np.where((y[1:-1]<y[:-2])&(y[1:-1]<y[2:]))[0]+1
ax.scatter(x[idx],y[idx],s=40,color=RED,zorder=5)
ax.set_xlabel('параметр (схематично)'); ax.set_ylabel('loss'); ax.set_title('Невыпуклая задача: стационарные точки уже не равнозначны'); ax.grid(alpha=.15)
fig.tight_layout(); save(fig,'nonconvex_loss_landscape')

print('generated', len(list(OUT.glob('*.pdf'))), 'pdf figures')
