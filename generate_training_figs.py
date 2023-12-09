#This plot generates figures for the training procedures

from visualization import simple_plot
import matplotlib.pyplot as plt
from util import load_pickle
from pathlib import Path

def make_plot(full_path_y, full_path_x, axes, x_lab, y_lab, title):
    """
    makes the plot on the given axes
    Args:
        full_path_y: (str) full path to the pickle file containing y-axis data
        full_path_x: (str) full path to the pickle file containing x-axis data
        x_lab: (str)
        y_lab: (str)
        title: (str)
        axes: (axes)

    Returns:
    """

    data_x = load_pickle(full_path_x)
    data_y = load_pickle(full_path_y)

    simple_plot(data_x, data_y, x_lab, y_lab, save=False, title=title, axes=axes,
                make_square=False, legend_keys = {"svd":"LSI", "lda":"LDA", "pmf":"PMF"})

def generate_plot(base_locations, metric, file_name, x_file_name):

    fig = plt.figure(figsize=(16, 4))
    axes_all = [fig.add_subplot(1, 4, i+1) for i in range(len(base_locations))]

    for i in range(len(base_locations)):
        loc = "output/"+ base_locations[i]
        print(loc)
        make_plot(loc + "/" + file_name, loc + "/" + x_file_name, axes_all[i],
                  "#topics", metric, " ".join(loc.split("_")[2:]).capitalize())

    folder_name = Path("training_figures")
    folder_name.mkdir(parents=True, exist_ok=True)
    output_loc = "{}/{}.pdf".format(folder_name, metric)
    fig.tight_layout()
    fig.savefig(output_loc, bbox_inches="tight")

if __name__ == "__main__":

    locations = ["test_output_chinese_policy", "test_output_english_policy",
                      "test_output_english_opinion", "test_output_chinese_opinion"]
    locations = sorted(locations)

    quantity1 = "perplexity" #"coherence_nmi"
    quantity2 = "coherence_umass"
    quantity3 = "coherence_nmi"
    topic_size_file_name = "topic_size.pkl"

    generate_plot(locations, quantity1, "{}.pkl".format(quantity1),
                  topic_size_file_name)
    generate_plot(locations, quantity2, "{}.pkl".format(quantity2),
                  topic_size_file_name)
    generate_plot(locations, quantity3, "{}.pkl".format(quantity3),
                  topic_size_file_name)



