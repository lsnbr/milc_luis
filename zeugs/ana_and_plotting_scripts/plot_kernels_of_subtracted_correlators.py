from import_and_plotting_tech import *





twopi = 2.0 * np.pi

def K(n, x):  # x = omega'/T
    return x / (x**2 + (twopi*n)**2)

def S01(x):
    return K(0, x) - K(1, x)

def S012(x):
    return K(0, x) - (4/3)*K(1, x) + (1/3)*K(2, x)

def xS01(x):
    return x * S01(x)

def xS012(x):
    return x * S012(x)

# linear x-range (now ok to include 0)
x_min, x_max = 0.01, 30.0
x = np.linspace(x_min, x_max, 2000)

fac = 0.75
fig, ax = plt.subplots(figsize=(fac*wlatex, fac*0.65*wlatex), constrained_layout=True)

ax.plot(x, xS01(x), label=r"$h^{(0,1)}$")
ax.plot(x, xS012(x), label=r"$h^{(0,1,2)}$")

ax.set_xlim(x_min, x_max)
ax.set_ylim(0, None)
ax.set_xlabel(r"$\omega'/T$")
ax.set_ylabel(r"kernel $h$")

ax.legend(frameon=True)

out = Path.cwd() / 'zeugs' / "plots" / "subtraction_kernels_x_linear.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out}")