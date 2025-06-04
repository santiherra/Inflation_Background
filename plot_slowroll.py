import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scienceplots

plt.style.use(['science', 'no-latex'])

x = np.linspace(-1, 12, 1000)
x1, x2 = 2.5, 11
point = np.array([10, 1.05])
V = ( 1 - np.exp( -np.sqrt(2/3)*x ) )**2  # Starobinsky potential, in units of M^4, and x = phi / M_P

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

ax.plot(x, V, color='blue', label=r'$V(\phi)$')
ax.arrow(point[0], point[1], -1, 0, head_width=0.06, head_length=0.15, fc='red', ec='red', label='Vector 1')
ax.arrow(point[0], point[1]-.15, -.45, 0, head_width=0.06, head_length=0.15, fc='black', ec='black')
ax.arrow(point[0], point[1]-.15, .45, 0, head_width=0.06, head_length=0.15, fc='black', ec='black')
ax.plot(point[0], point[1], 'ro')
ax.text(point[0] - .35, point[1] + .1, r'$\phi_0$', fontsize=16)
ax.text(point[0] - .35, point[1] - .3, r'$\delta \phi$', fontsize=16)
ax.text(0.3, 1.5, r'$V(\phi)$', fontsize=16)
ax.text(12, 0.1, r'$\phi$', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-1.6, 12.6)
ax.set_ylim(-.1, 1.7)
ax.annotate("", xy=(12.6, 0), xytext=(-1.6, 0),
            arrowprops=dict(arrowstyle="->", linewidth=1))
ax.annotate("", xy=(0, 1.7), xytext=(0, -0.1),
            arrowprops=dict(arrowstyle="->", linewidth=1))
ax.axvspan(x1, x2, color='lime', alpha=0.2)
ax.axvline(x = x1, linestyle='--', color='green', linewidth=.5)
ax.axvline(x = x2, linestyle='--', color='green', linewidth=.5)

plt.show()
