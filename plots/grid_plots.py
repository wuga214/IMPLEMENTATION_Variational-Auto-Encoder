import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
sns.axes_style("white")


def show_samples(images, row, col, name="Unknown", save=True):
    num_images = row*col
    fig = plt.figure(figsize=(col, row))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(row, col),
                     axes_pad=0.)
    for i in xrange(num_images):
        im = images[i].reshape(28, 28)
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im)
    plt.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig('figs/train/grid/'+name+'.pdf', bbox_inches="tight", pad_inches=0, format='pdf')
    else:
        plt.show()


#From some github code
def show_latent_scatter(vae, data, name="latent"):
    n_test = 5000
    batch_size = 100
    zs = np.zeros((n_test, 2), dtype=np.float32)
    labels = np.zeros(n_test)
    for i in range(int(n_test / batch_size)):
        x, y = data.test.next_batch(batch_size)
        labels[(100 * i):(100 * (i + 1))] = y
        z = vae.x2z(x)
        zs[(100 * i):(100 * (i + 1)), :] = z

    indices = np.array([np.where(labels == i)[0] for i in range(10)])
    classes = np.array([zs[index] for index in indices])
    means = np.array([np.mean(c, axis=0) for c in classes])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(zs[:, 0], zs[:, 1], c=labels)

    # annotate means
    for i, mean in enumerate(means):
        ax.annotate(str(i), xy=mean, size=16,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    # plot details
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    plt.savefig('figs/train/scatter/' + name + '.pdf', format='pdf')
