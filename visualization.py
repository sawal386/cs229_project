import matplotlib.pyplot as plt
import numpy

from sklearn.manifold import TSNE
from pathlib import Path

def visualize_embeddings(embed_mat, n=50, word_index_map=None, save=True,
                         save_path="images", save_name="embedding", title=""):
    """
    visualize the word embeddings using TSNE
    Args:
        embed_mat: (np.ndarray) the embedding matrix. shape: (dim, n_words)
        n: (int) number of embeddings to visualize
        word_index_map: (dict) (int) index -> (str) word
        save: (bool) whether or to save the image or not
        save_path: (str) the path where the image is saved
        save_name: (str) the name of the image
        title: (str) title of the plot
    """

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_label("dim 1")
    axes.set_ylabel("dim 2")
    axes.set_title(title)

    if n > embed_mat.shape[1]:
        n = embed_mat.shape[1]
    tsne_embded = TSNE(n_components=2).fit_transform(embed_mat)[:n]
    print(tsne_embded.shape)
    axes.scatter(tsne_embded[:, 0], tsne_embded[:, 1])
    if save:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        full_path = save_path / "{}.pdf".format(save_name)
        fig.savefig(full_path)


def simple_plot(x_dict, y_dict, x_lab, y_lab, save=True, save_path="figures",
                save_name="sample_figure", legend_on=True):
    """
    makes a simple plot
    Args:
        x_dict: (dict) (str/int) name -> (np.ndarray) x values
        y_dict: (dict) (str/int) name -> (np.ndarray) y values
        x_lab: (str) x-axis label
        y_lab: (str) y-axis label
        save: (bool) whether to save the plot or not
        save_path: (str) path to the folder
        save_name: (str) name of the file
        legend_on: (bool) is legend used or not
    """
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_ylabel(y_lab)
    axes.set_xlabel(x_lab)
    for key in x_dict:
        x, y = x_dict[key], y_dict[key]
        axes.plot(x, y)

    if legend_on:
        axes.legend()

    if save:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if ".pdf" not in save_name:
            save_name = save_name + ".pdf"
        full_path = save_path / "{}".format(save_name)
        fig.savefig(full_path)

