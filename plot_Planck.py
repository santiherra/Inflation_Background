import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scienceplots

plt.style.use(['science', 'no-latex'])

# CImport image with contours
img = mpimg.imread('contours_Planck_2.jpg') 

# Real axes limits
ns_min, ns_max = 0.945, 1
r_min, r_max = 0.0, 0.26

fig, ax = plt.subplots(1,1, figsize=(6, 4), dpi=200)

# Shows image in real range
ax.imshow(img, extent=[ns_min, ns_max, r_min, r_max], aspect='auto')

# Our point
ns_model = .9621
r_model = .0173
ax.plot(ns_model, r_model, color='yellow', marker='*', markersize=8)

ax.set_xlabel(r'$n_s$', fontsize=15)
ax.set_ylabel(r'$r$', fontsize=15)

plt.tight_layout()
plt.show()



