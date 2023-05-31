import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"


fig, ax = plt.subplots()
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$\\beta$')
ax.set_title(r'$\mathrm{Inverse\ Gamma\ Distribution}$')


x = [1,2,3,4]
y = [1,2,3,4]

# Plot and display your figure
plt.plot(x, y)
plt.show()