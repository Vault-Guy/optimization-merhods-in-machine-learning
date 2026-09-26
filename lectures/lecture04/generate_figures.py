import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
RED = '#E52E5A'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'mathtext.fontset': 'dejavusans',
    'font.family': 'DejaVu Sans',
})

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.06)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.06, dpi=170)
    plt.close(fig)

# 1. Conditioning contours kappa 1/10/100
fig, axs = plt.subplots(1, 3, figsize=(11.8, 3.3))
for ax, k in zip(axs, [1, 10, 100]):
    x = np.linspace(-4.5, 4.5, 400)
    y = np.linspace(-4.5, 4.5, 400)
    X,Y = np.meshgrid(x,y)
    Z = 0.5*(X**2 + k*Y**2)
    ax.contour(X,Y,Z,levels=[0.5,1.5,3.0,6.0], linewidths=1.4)
    ax.set_aspect('equal')
    ax.axhline(0,lw=.5,alpha=.45); ax.axvline(0,lw=.5,alpha=.45)
    ax.set_xlim(-4.2,4.2); ax.set_ylim(-4.2,4.2)
    ax.set_title(rf'$\kappa={k}$')
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Чем больше число обусловленности, тем сильнее различается кривизна по направлениям', y=1.03)
fig.tight_layout()
save(fig,'conditioning_contours')

# 2. Sensitivity solution comparison: readable infographic
fig, ax = plt.subplots(figsize=(10.4, 4.0))
ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def card(y0, kappa, rel_x, accent=False):
    # outer card
    rect = plt.Rectangle((0.03, y0), 0.94, 0.32, fill=False, lw=1.2)
    ax.add_patch(rect)
    ax.text(0.07, y0+0.235, rf'$\kappa={kappa}$', fontsize=15, weight='bold', va='center')
    ax.text(0.23, y0+0.235, 'одинаковое возмущение правой части', fontsize=11, va='center')
    ax.text(0.23, y0+0.145, r'$\|\Delta b\|/\|b\|=10\%$', fontsize=15, va='center')
    ax.annotate('', xy=(0.68, y0+0.16), xytext=(0.57, y0+0.16),
                arrowprops=dict(arrowstyle='->', lw=2.0))
    ax.text(0.72, y0+0.235, 'изменение решения', fontsize=11, va='center')
    ax.text(0.72, y0+0.135, rf'$\|\Delta x\|/\|x\|={rel_x}$',
            fontsize=17 if accent else 15, weight='bold' if accent else 'normal', va='center')

card(0.57, 1, r'10\%', False)
card(0.13, 100, r'1000\%', True)
ax.text(0.5, 0.025, 'В чувствительном направлении усиление относительной ошибки может достигать числа обусловленности.',
        ha='center', fontsize=11.2)
fig.tight_layout()
save(fig,'sensitivity_solution')

# 3. Why the linear LOCAL model needs a quadratic penalty
fig, axs = plt.subplots(1,2,figsize=(10.3,3.4))
h=np.linspace(-4,4,400); g=1.8
axs[0].plot(h,g*h,lw=2)
axs[0].axhline(0,lw=.6,alpha=.4); axs[0].axvline(0,lw=.6,alpha=.4)
axs[0].set_title('Линейная локальная модель $m_{lin}(h)$')
axs[0].set_xlabel('$h$'); axs[0].set_ylabel(r'$m_{lin}(h)=\nabla f(x)^\top h$')
axs[0].annotate(r'$h=-t\nabla f(x),\ t\to\infty$',xy=(-3.0,-5.4),xytext=(-1.6,-2.4),arrowprops=dict(arrowstyle='->'))
axs[0].text(-3.65,5.7,r'$m_{lin}(-t\nabla f)=-t\|\nabla f\|^2\to-\infty$',fontsize=9.5)
alpha=0.8; m=g*h + h**2/(2*alpha); hstar=-alpha*g
axs[1].plot(h,m,lw=2); axs[1].scatter([hstar],[g*hstar+hstar**2/(2*alpha)],s=55,zorder=3)
axs[1].axhline(0,lw=.6,alpha=.4); axs[1].axvline(0,lw=.6,alpha=.4)
axs[1].set_title(r'Штраф $\|h\|^2/(2\alpha)$ делает модель ограниченной')
axs[1].set_xlabel('$h$'); axs[1].set_ylabel('$m_x(h)$')
axs[1].annotate(r'$h^*=-\alpha\nabla f(x)$',xy=(hstar,g*hstar+hstar**2/(2*alpha)),xytext=(0.0,-1.0),arrowprops=dict(arrowstyle='->'))
fig.tight_layout(); save(fig,'regularized_local_model')

# Helpers
def gd_path(A,b,x0,alpha,n=30):
    xs=[np.array(x0,dtype=float)]; x=np.array(x0,dtype=float)
    for _ in range(n):
        x=x-alpha*(A@x-b); xs.append(x.copy())
    return np.array(xs)

def quad_contour(ax,A,b=np.zeros(2),path=None,title=''):
    xx=np.linspace(-5,5,500); yy=np.linspace(-4,4,500); X,Y=np.meshgrid(xx,yy)
    pts=np.stack([X,Y],axis=-1)
    Z=0.5*np.einsum('...i,ij,...j->...',pts,A,pts)-np.einsum('...i,i->...',pts,b)
    zmin=np.nanmin(Z); levels=zmin+np.geomspace(0.15,30,10)
    ax.contour(X,Y,Z,levels=levels,linewidths=1.0,alpha=.85)
    if path is not None:
        ax.plot(path[:,0],path[:,1],'-o',ms=2.8,lw=1.6)
        ax.scatter(path[0,0],path[0,1],s=35,marker='s'); ax.scatter(path[-1,0],path[-1,1],s=45,marker='*')
    ax.set_aspect('equal'); ax.set_xlim(-4.8,4.8); ax.set_ylim(-3.8,3.8); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)

# 4. GD paths conditioning comparison
fig,axs=plt.subplots(1,3,figsize=(11.6,3.4))
for ax,k in zip(axs,[1,10,100]):
    A=np.diag([1.0,k]); path=gd_path(A,np.zeros(2),[-4,2.7],1/k,n=25)
    quad_contour(ax,A,path=path,title=rf'$\kappa={k}$, $\alpha=1/L$')
fig.suptitle('Один и тот же алгоритм ведёт себя всё медленнее при росте обусловленности',y=1.02)
fig.tight_layout(); save(fig,'gd_condition_paths')

# 5. Scalar recurrence regimes
fig,axs=plt.subplots(1,3,figsize=(11.1,3.2))
regimes=[(0.65,r'$q_i=0.65$: монотонное затухание'),(-0.65,r'$q_i=-0.65$: колебательное затухание'),(-1.15,r'$q_i=-1.15$: расходимость')]
for ax,(q,title) in zip(axs,regimes):
    kk=np.arange(13); z=q**kk
    ax.axhline(0,lw=.7,alpha=.4); ax.plot(kk,z,'o-',lw=1.8,ms=4)
    ax.set_title(title,fontsize=10); ax.set_xlabel('$k$'); ax.set_ylabel('$z_{k,i}$'); ax.grid(alpha=.18)
fig.tight_layout(); save(fig,'scalar_recurrence_regimes')

# 6. Stability interval for a concrete eigenvalue lambda=3
fig,ax=plt.subplots(figsize=(8.2,4.0)); lam=3.0
alpha=np.linspace(0,1.02,600); q=np.abs(1-alpha*lam)
ax.plot(alpha,q,lw=2.2,label=r'$|1-3\alpha|$')
ax.axhline(1,ls='--',lw=1.1)
ax.axvline(1/lam,ls=':',lw=1.2)
ax.axvline(2/lam,ls='--',lw=1.2)
ax.fill_between(alpha,0,q,where=((alpha>0)&(alpha<2/lam)&(q<1)),alpha=.14)
ax.scatter([1/lam],[0],s=48,zorder=5)
ax.scatter([2/lam],[1],s=48,zorder=5)
ax.annotate(r'$\alpha=1/3$: $q=0$',xy=(1/lam,0),xytext=(8,18),textcoords='offset points',fontsize=10)
ax.annotate(r'$\alpha=2/3$: $q=-1$',xy=(2/lam,1),xytext=(-75,18),textcoords='offset points',fontsize=10)
ax.text(0.19,0.33,'сходимость',fontsize=11)
ax.text(0.79,1.55,'расходимость',fontsize=11)
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$|1-3\alpha|$'); ax.set_ylim(0,2.15); ax.set_xlim(0,1.0)
ax.set_xticks([0,1/3,2/3,1.0],['0',r'$1/3$',r'$2/3$','1'])
ax.set_title(r'Пример $\lambda=3$: сходимость требует $|1-3\alpha|<1$')
ax.grid(alpha=.18); ax.legend(fontsize=9,loc='upper left')
fig.tight_layout(); save(fig,'stability_interval')

# 7. contraction vs kappa
fig,ax=plt.subplots(figsize=(7.6,3.8)); kap=np.logspace(0,3,400); q=1-1/kap
ax.plot(kap,q,lw=2); ax.set_xscale('log'); ax.set_xlabel(r'$\kappa$'); ax.set_ylabel(r'$q=1-1/\kappa$')
ax.set_title(r'При шаге $\alpha=1/L$: чем хуже обусловленность, тем ближе $q$ к 1'); ax.grid(alpha=.2,which='both')
for kk in [2,10,100]:
    ax.scatter([kk],[1-1/kk],s=35); ax.annotate(rf'$\kappa={kk}$',xy=(kk,1-1/kk),xytext=(6,-18 if kk==2 else 8),textcoords='offset points')
fig.tight_layout(); save(fig,'contraction_vs_kappa')

# 7b. Error decay for representative condition numbers
fig,ax=plt.subplots(figsize=(8.2,4.0))
steps=np.arange(0,121)
for kap in [1,10,100]:
    q=1-1/kap
    decay=np.zeros_like(steps,dtype=float) if kap==1 else q**steps
    if kap==1:
        decay[0]=1.0
    ax.semilogy(steps, np.maximum(decay,1e-6), lw=2, label=rf'$\kappa={kap}$, $q={q:.2f}$')
ax.set_xlabel('итерация $k$'); ax.set_ylabel(r'верхняя оценка $q^k$')
ax.set_title(r'При $\alpha=1/L$: большое $\kappa$ означает медленное сокращение ошибки')
ax.grid(alpha=.2,which='both'); ax.legend(fontsize=9)
fig.tight_layout(); save(fig,'error_decay_conditioning')

# 8. nonquadratic rapid gradient change
fig,axs=plt.subplots(1,2,figsize=(10.0,3.5)); x=np.linspace(-2.5,2.5,500)
f1=0.5*x**2; f2=np.exp(1.7*x)/6
axs[0].plot(x,f1,lw=2); axs[0].set_title('Умеренная кривизна'); axs[0].set_xlabel('$x$'); axs[0].set_ylabel('$f(x)$'); axs[0].grid(alpha=.18)
axs[1].plot(x,f2,lw=2); axs[1].set_title('Градиент меняется очень быстро'); axs[1].set_xlabel('$x$'); axs[1].grid(alpha=.18)
for xx in [-1,0,1]:
    y=np.exp(1.7*xx)/6; gg=1.7*np.exp(1.7*xx)/6; dx=.35
    axs[1].plot([xx-dx,xx+dx],[y-gg*dx,y+gg*dx],lw=1.0,alpha=.7)
fig.tight_layout(); save(fig,'gradient_change_nonquadratic')

# 9. smooth upper quadratic model
fig,ax=plt.subplots(figsize=(7.8,4.0)); x=np.linspace(-1.8,2.2,500)
f=np.log1p(np.exp(2*x))/2 + 0.08*x**2; x0=-0.35
sig=1/(1+np.exp(-2*x0)); gg=sig + 0.16*x0; L=1.2
f0=np.log1p(np.exp(2*x0))/2 + 0.08*x0**2
upper=f0+gg*(x-x0)+0.5*L*(x-x0)**2; lin=f0+gg*(x-x0)
ax.plot(x,f,lw=2,label='$f(x)$'); ax.plot(x,lin,ls='--',lw=1.5,label='линейная модель'); ax.plot(x,upper,lw=1.7,label='квадратичная верхняя модель')
ax.scatter([x0],[f0],s=45,zorder=5); ax.set_xlabel('$x$'); ax.set_ylabel('значение'); ax.set_title('Гладкость даёт квадратичную верхнюю оценку')
ax.legend(fontsize=9); ax.grid(alpha=.18); fig.tight_layout(); save(fig,'smooth_upper_bound')

# 10. Step sizes same quadratic
fig,axs=plt.subplots(1,4,figsize=(13.2,3.1)); A=np.diag([1.,10.]); L=10.; x0=[-4,2.6]
for ax,c in zip(axs,[0.2,1.0,1.8,2.2]):
    path=gd_path(A,np.zeros(2),x0,c/L,n=22); quad_contour(ax,A,path=path,title=rf'$\alpha={c}/L$')
fig.suptitle('Шаг управляет компромиссом: скорость, колебания и устойчивость',y=1.02); fig.tight_layout(); save(fig,'step_size_paths')

# 11. q(alpha) optimal: show both endpoint factors and their maximum
fig,ax=plt.subplots(figsize=(8.4,4.2)); mu=1.0; L=10.0
alpha=np.linspace(0,2/L,700)
q_mu=np.abs(1-alpha*mu)
q_L=np.abs(1-alpha*L)
q=np.maximum(q_mu,q_L)
aopt=2/(L+mu); qopt=(L-mu)/(L+mu)
ax.plot(alpha,q_mu,ls='--',lw=1.6,label=r'$|1-\alpha\mu|$')
ax.plot(alpha,q_L,ls='--',lw=1.6,label=r'$|1-\alpha L|$')
ax.plot(alpha,q,lw=2.6,label=r'$q(\alpha)=\max\{\cdot,\cdot\}$')
ax.scatter([aopt],[qopt],s=58,zorder=5)
ax.axvline(1/L,ls=':',lw=1.2,label=r'$1/L$')
ax.axvline(aopt,ls='-.',lw=1.2,label=r'$2/(L+\mu)$')
ax.annotate(r'минимаксный оптимум',xy=(aopt,qopt),xytext=(-85,-35),textcoords='offset points',arrowprops=dict(arrowstyle='->',lw=1),fontsize=10)
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('множитель сокращения')
ax.set_title(r'$\mu=1,\ L=10$: оптимальный шаг уравнивает два худших направления')
ax.legend(fontsize=8.5,loc='upper center',ncol=2); ax.grid(alpha=.18)
fig.tight_layout(); save(fig,'optimal_alpha')

# 12. Compare alpha choices error decay
fig,ax=plt.subplots(figsize=(7.8,4.0)); A=np.diag([1.,10.]); x0=np.array([-4.,2.6]); b=np.zeros(2)
for alpha,label in [(1/10,r'$1/L$'),(2/11,r'$2/(L+\mu)$')]:
    path=gd_path(A,b,x0,alpha,n=30); err=np.linalg.norm(path,axis=1); ax.semilogy(err,lw=2,label=label)
ax.set_xlabel('итерация $k$'); ax.set_ylabel(r'$\|x_k-x^*\|$'); ax.set_title('Оптимальный фиксированный шаг заметно ускоряет сходимость')
ax.legend(); ax.grid(alpha=.2,which='both'); fig.tight_layout(); save(fig,'alpha_error_decay')

# 13. descent step visual on a globally L-smooth function
fig,ax=plt.subplots(figsize=(7.8,4.1)); x=np.linspace(-2.0,2.4,500)
# f(x)=0.5 log(1+e^{2x})+0.08x^2 has f''(x)<=0.66, so L=0.7 is valid globally
f=np.log1p(np.exp(2*x))/2 + 0.08*x**2
x0=1.15
sig=1/(1+np.exp(-2*x0)); gg=sig+0.16*x0
L=0.7; alpha=1/L; x1=x0-alpha*gg
f0=np.log1p(np.exp(2*x0))/2 + 0.08*x0**2
f1=np.log1p(np.exp(2*x1))/2 + 0.08*x1**2
upper=f0+gg*(x-x0)+0.5*L*(x-x0)**2
ax.plot(x,f,lw=2,label='$f$'); ax.plot(x,upper,lw=1.6,label='верхняя квадратичная модель')
ax.scatter([x0,x1],[f0,f1],s=55,zorder=5)
ax.annotate('$x_k$',xy=(x0,f0),xytext=(8,10),textcoords='offset points')
ax.annotate('$x_{k+1}$',xy=(x1,f1),xytext=(-22,-20),textcoords='offset points')
ax.annotate('',xy=(x1,f1),xytext=(x0,f0),arrowprops=dict(arrowstyle='->',lw=1.7,color=RED))
ax.set_title(r'При $\alpha=1/L$ шаг уменьшает верхнюю модель и саму функцию'); ax.set_xlabel('$x$'); ax.grid(alpha=.18); ax.legend(fontsize=9)
fig.tight_layout(); save(fig,'descent_step_visual')

# 14. Lipschitz gradient visualized as a cone around one known gradient value
fig, axs = plt.subplots(1,2,figsize=(10.6,3.8), sharey=True)
t=np.linspace(-2.6,2.6,500)
g=np.sin(t)  # gradient of f(t)=-cos(t), globally 1-Lipschitz
x0=0.0; g0=0.0
for ax, Lcand, title in [(axs[0],1.0,r'$L=1$: условие выполняется'),
                          (axs[1],0.4,r'$L=0.4$: конус слишком узкий')]:
    upper=g0+Lcand*np.abs(t-x0)
    lower=g0-Lcand*np.abs(t-x0)
    ax.fill_between(t,lower,upper,alpha=.12,label=r'$|g(t)-g(x_0)|\leq L|t-x_0|$')
    ax.plot(t,g,lw=2,label=r"$g(t)=f'(t)=\sin t$")
    ax.plot(t,upper,ls='--',lw=1.2)
    ax.plot(t,lower,ls='--',lw=1.2)
    ax.scatter([x0],[g0],s=45,zorder=5)
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('$t$'); ax.grid(alpha=.18)
axs[0].set_ylabel("градиент $g(t)=f'(t)$")
fig.suptitle(r'Липшицевость градиента: его график не должен выходить из конуса наклона $\pm L$ вокруг известной точки',y=1.02)
axs[0].legend(fontsize=8,loc='upper left')
fig.tight_layout(); save(fig,'smoothness_gradient_change')

print('generated', len(list(OUT.glob('*.pdf'))), 'pdf figures')

# --- Revised figures after lecture review ---

# Sensitivity: one shared perturbation, two very different solution responses.
fig = plt.figure(figsize=(10.6, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.35], hspace=0.55, wspace=0.34)

# Same perturbation in b-space
ax = fig.add_subplot(gs[0, :])
b = np.array([1.0, 0.0])
db = np.array([0.0, 0.1])
b2 = b + db
ax.set_aspect('equal')
ax.arrow(0, 0, b[0], b[1], width=0.004, head_width=0.045, head_length=0.06, length_includes_head=True)
ax.arrow(0, 0, b2[0], b2[1], width=0.004, head_width=0.045, head_length=0.06, length_includes_head=True)
ax.plot([b[0], b2[0]], [b[1], b2[1]], linestyle='--', linewidth=1.2)
ax.text(1.03, 0.00, r'$b$', va='center', fontsize=12)
ax.text(1.03, 0.105, r'$b+\Delta b$', va='center', fontsize=12)
ax.text(0.42, 0.135, r'$\|\Delta b\|/\|b\|=10\%$', fontsize=12)
ax.set_xlim(-0.05, 1.35); ax.set_ylim(-0.08, 0.23)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Одно и то же возмущение правой части в обеих системах')
for s in ax.spines.values(): s.set_visible(False)

# Well-conditioned system
ax1 = fig.add_subplot(gs[1, 0])
x = np.array([1.0, 0.0]); dx1 = np.array([0.0, 0.1]); xp1 = x + dx1
ax1.arrow(0,0,x[0],x[1],width=0.004,head_width=0.05,head_length=0.07,length_includes_head=True)
ax1.arrow(0,0,xp1[0],xp1[1],width=0.004,head_width=0.05,head_length=0.07,length_includes_head=True)
ax1.plot([x[0],xp1[0]],[x[1],xp1[1]],linestyle='--',linewidth=1.2)
ax1.text(0.05,0.78,r'$A_1=I,\ \kappa(A_1)=1$',transform=ax1.transAxes,fontsize=12)
ax1.text(0.05,0.66,r'$x=A_1^{-1}b=(1,0)$',transform=ax1.transAxes,fontsize=11)
ax1.text(0.05,0.54,r'$\Delta x=(0,0.1)$',transform=ax1.transAxes,fontsize=11)
ax1.text(0.05,0.40,r'$\|\Delta x\|/\|x\|=10\%$',transform=ax1.transAxes,fontsize=13,weight='bold')
ax1.set_xlim(-0.05,1.3); ax1.set_ylim(-0.08,0.28); ax1.set_aspect('equal')
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title('Хорошо обусловленная система')
for s in ax1.spines.values(): s.set_visible(False)

# Ill-conditioned system
ax2 = fig.add_subplot(gs[1, 1])
dx2 = np.array([0.0, 10.0]); xp2 = x + dx2
ax2.arrow(0,0,x[0],x[1],width=0.035,head_width=0.28,head_length=0.45,length_includes_head=True)
ax2.arrow(0,0,xp2[0],xp2[1],width=0.035,head_width=0.28,head_length=0.45,length_includes_head=True)
ax2.plot([x[0],xp2[0]],[x[1],xp2[1]],linestyle='--',linewidth=1.2)
ax2.text(0.05,0.78,r'$A_2=\mathrm{diag}(1,0.01),\ \kappa(A_2)=100$',transform=ax2.transAxes,fontsize=11.5)
ax2.text(0.05,0.66,r'$x=A_2^{-1}b=(1,0)$',transform=ax2.transAxes,fontsize=11)
ax2.text(0.05,0.54,r'$\Delta x=(0,10)$',transform=ax2.transAxes,fontsize=11)
ax2.text(0.05,0.40,r'$\|\Delta x\|/\|x\|=1000\%$',transform=ax2.transAxes,fontsize=13,weight='bold')
ax2.set_xlim(-0.25,2.25); ax2.set_ylim(-0.7,11.5); ax2.set_aspect('auto')
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title('Плохо обусловленная система')
for s in ax2.spines.values(): s.set_visible(False)

fig.tight_layout()
save(fig, 'sensitivity_solution')

# Linear model: the unboundedness belongs to the local affine model, not to f.
fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.7))
h = np.linspace(-2.4, 2.4, 500)
ftrue = (h + 1.0)**2
lin = 1.0 + 2.0*h
axs[0].plot(h, ftrue, linewidth=2, label=r'истинная $f(x+h)$')
axs[0].plot(h, lin, linestyle='--', linewidth=2, label='линейная модель')
axs[0].axvspan(-0.45, 0.45, alpha=0.10, label='локальная область')
axs[0].scatter([0],[1],s=45,zorder=5)
axs[0].axhline(0,linewidth=.6,alpha=.4); axs[0].axvline(0,linewidth=.6,alpha=.4)
axs[0].set_ylim(-4.2, 8.5)
axs[0].set_xlabel('$h$')
axs[0].set_title('Линейная аппроксимация надёжна только рядом с $h=0$')
axs[0].legend(fontsize=8, loc='upper left')
axs[0].grid(alpha=.16)

alpha = 0.25
g = 2.0
m = g*h + h**2/(2*alpha)
hstar = -alpha*g
axs[1].plot(h, g*h, linestyle='--', linewidth=1.6, label=r'$m_{lin}(h)$')
axs[1].plot(h, m, linewidth=2.1, label=r'$m_x(h)=m_{lin}(h)+\|h\|^2/(2\alpha)$')
axs[1].scatter([hstar], [g*hstar+hstar**2/(2*alpha)], s=55, zorder=5)
axs[1].annotate(r'$h^*=-\alpha\nabla f(x)$', xy=(hstar,g*hstar+hstar**2/(2*alpha)), xytext=(0.0,-1.35), arrowprops=dict(arrowstyle='->'))
axs[1].axhline(0,linewidth=.6,alpha=.4); axs[1].axvline(0,linewidth=.6,alpha=.4)
axs[1].set_xlabel('$h$')
axs[1].set_title('Квадратичный штраф задаёт конечную длину шага')
axs[1].legend(fontsize=8, loc='upper left')
axs[1].grid(alpha=.16)
fig.tight_layout()
save(fig, 'regularized_local_model')

# Worst direction for alpha=1/L.
fig, ax = plt.subplots(figsize=(5.0, 3.15))
r = np.linspace(0, 1, 300)
q = 1-r
ax.plot(r, q, linewidth=2.2)
mu_over_L = 0.2
ax.scatter([mu_over_L, 1.0], [1-mu_over_L, 0], s=50, zorder=4)
ax.axvline(mu_over_L, linestyle='--', linewidth=1.0, alpha=.65)
ax.text(mu_over_L+0.02, 0.84, r'$\lambda=\mu$', fontsize=10.5)
ax.text(mu_over_L+0.02, 0.73, r'$q_{max}=1-\mu/L$', fontsize=10.5)
ax.text(0.78, 0.10, r'$\lambda=L$', fontsize=10.5)
ax.set_xlim(0,1.02); ax.set_ylim(-0.04,1.02)
ax.set_xlabel(r'$\lambda_i/L$')
ax.set_ylabel(r'$q_i=1-\lambda_i/L$')
ax.set_title('Меньшая кривизна $\Rightarrow$ большая доля ошибки остаётся')
ax.grid(alpha=.18)
fig.tight_layout()
save(fig, 'worst_direction_factor')

# Lipschitz gradient: same horizontal displacement, different gradient changes.
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.65), sharex=True)
t = np.linspace(-0.8, 0.9, 300)
t1, t2 = -0.35, 0.45
for ax, slope, title in [(axs[0],1.0,r'градиент меняется умеренно: $L=1$'),
                         (axs[1],4.0,r'градиент меняется быстро: $L=4$')]:
    g = slope*t
    g1, g2 = slope*t1, slope*t2
    ax.plot(t,g,linewidth=2)
    ax.scatter([t1,t2],[g1,g2],s=48,zorder=5)
    ax.plot([t1,t2],[g1,g1],linestyle='--',linewidth=1.0)
    ax.plot([t2,t2],[g1,g2],linestyle='--',linewidth=1.0)
    ax.annotate(r'$\Delta t$', xy=((t1+t2)/2,g1), xytext=(0,-18), textcoords='offset points', ha='center')
    ax.annotate(r'$\Delta g$', xy=(t2,(g1+g2)/2), xytext=(8,0), textcoords='offset points', va='center')
    ax.text(0.05,0.90,rf'$|\Delta g|/|\Delta t|={slope:g}$',transform=ax.transAxes,fontsize=12,weight='bold')
    ax.axhline(0,linewidth=.6,alpha=.4); ax.axvline(0,linewidth=.6,alpha=.4)
    ax.set_title(title)
    ax.set_xlabel('$t$')
    ax.grid(alpha=.16)
axs[0].set_ylabel(r"$g(t)=f'(t)$")
fig.suptitle(r"$L$ ограничивает наклон любой секущей графика градиента $g=f'$", y=1.02)
fig.tight_layout()
save(fig, 'smoothness_gradient_change')

# -----------------------------------------------------------------------------
# Final layout-cleanup overrides (v5)
# -----------------------------------------------------------------------------

# Sensitivity comparison: clean geometry, all explanatory text outside arrows.
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.35))
# well-conditioned
ax = axs[0]
x = np.array([1.0, 0.0]); xp = np.array([1.0, 0.1])
ax.plot([0, x[0]], [0, x[1]], lw=2.4, label=r'$x$')
ax.plot([0, xp[0]], [0, xp[1]], lw=2.4, label=r'$x+\Delta x$')
ax.scatter([x[0], xp[0]], [x[1], xp[1]], s=42, zorder=4)
ax.plot([x[0], xp[0]], [x[1], xp[1]], ls='--', lw=1.2)
ax.set_xlim(-0.05, 1.22); ax.set_ylim(-0.06, 0.22); ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=.12)
ax.set_title(r'$A_1=I$, $\kappa(A_1)=1$', pad=8)
ax.text(0.5, -0.20, r'$\|\Delta x\|/\|x\|=10\%$', transform=ax.transAxes,
        ha='center', fontsize=13, weight='bold')
ax.legend(loc='upper left', fontsize=9, frameon=False)

# ill-conditioned
ax = axs[1]
x = np.array([1.0, 0.0]); xp = np.array([1.0, 10.0])
ax.plot([0, x[0]], [0, x[1]], lw=2.4, label=r'$x$')
ax.plot([0, xp[0]], [0, xp[1]], lw=2.4, label=r'$x+\Delta x$')
ax.scatter([x[0], xp[0]], [x[1], xp[1]], s=42, zorder=4)
ax.plot([x[0], xp[0]], [x[1], xp[1]], ls='--', lw=1.2)
ax.set_xlim(-0.25, 1.35); ax.set_ylim(-0.65, 10.9); ax.set_xticks([]); ax.set_yticks([])
ax.grid(alpha=.12)
ax.set_title(r'$A_2=\mathrm{diag}(1,0.01)$, $\kappa(A_2)=100$', pad=8)
ax.text(0.5, -0.20, r'$\|\Delta x\|/\|x\|=1000\%$', transform=ax.transAxes,
        ha='center', fontsize=13, weight='bold')
ax.legend(loc='upper left', fontsize=9, frameon=False)

fig.suptitle(r'Одно и то же относительное возмущение правой части: $\|\Delta b\|/\|b\|=10\%$', y=1.02)
fig.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.24, wspace=0.30)
save(fig, 'sensitivity_solution')

# Local linear model vs regularized local model: no labels placed on top of curves.
fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.15))
h = np.linspace(-2.4, 2.4, 500)
ftrue = (h + 1.0)**2
lin = 1.0 + 2.0*h
axs[0].plot(h, ftrue, lw=2.2, label=r'$f(x+h)$')
axs[0].plot(h, lin, ls='--', lw=2.0, label=r'$m_{\rm lin}(h)$')
axs[0].axvspan(-0.42, 0.42, alpha=0.10)
axs[0].scatter([0], [1], s=42, zorder=5)
axs[0].axhline(0, lw=.6, alpha=.35); axs[0].axvline(0, lw=.6, alpha=.35)
axs[0].set_ylim(-4.2, 8.5); axs[0].set_xlabel('$h$')
axs[0].set_title('Линейная модель хороша только локально')
axs[0].legend(fontsize=8.5, loc='upper left', frameon=False)
axs[0].grid(alpha=.14)

alpha = 0.25; g = 2.0
m = g*h + h**2/(2*alpha); hstar = -alpha*g
axs[1].plot(h, g*h, ls='--', lw=1.8, label=r'$m_{\rm lin}(h)$')
axs[1].plot(h, m, lw=2.2, label=r'$m_x(h)$')
axs[1].scatter([hstar], [g*hstar + hstar**2/(2*alpha)], s=55, zorder=5)
axs[1].axhline(0, lw=.6, alpha=.35); axs[1].axvline(0, lw=.6, alpha=.35)
axs[1].set_xlabel('$h$'); axs[1].set_title('Квадратичный штраф создаёт конечный минимум')
axs[1].legend(fontsize=8.5, loc='upper left', frameon=False)
axs[1].grid(alpha=.14)
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.18, wspace=0.26)
save(fig, 'regularized_local_model')

# Lipschitz-gradient picture: clean secant comparison, no in-plot delta labels.
fig, axs = plt.subplots(1, 2, figsize=(10.1, 3.15), sharex=True)
t = np.linspace(-0.8, 0.9, 300)
t1, t2 = -0.35, 0.45
for ax, slope, title in [
    (axs[0], 1.0, r'$L=1$: градиент меняется умеренно'),
    (axs[1], 4.0, r'$L=4$: градиент меняется быстрее')
]:
    gvals = slope*t
    g1, g2 = slope*t1, slope*t2
    ax.plot(t, gvals, lw=2.2)
    ax.scatter([t1, t2], [g1, g2], s=48, zorder=5)
    ax.plot([t1, t2], [g1, g2], lw=2.0, ls='--')
    ax.axhline(0, lw=.6, alpha=.35); ax.axvline(0, lw=.6, alpha=.35)
    ax.set_title(title, pad=8)
    ax.set_xlabel('$t$')
    ax.text(0.05, 0.88, rf'наклон секущей $={slope:g}$', transform=ax.transAxes,
            fontsize=11.5, weight='bold')
    ax.grid(alpha=.14)
axs[0].set_ylabel(r"$g(t)=f'(t)$")
fig.suptitle(r'Для $L$-липшицева градиента модуль наклона любой секущей графика $g=f\prime$ не превосходит $L$', y=1.03)
fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.18, wspace=0.24)
save(fig, 'smoothness_gradient_change')