import matplotlib.pyplot as plt


class MetricsPlotter(object):
    def __init__(self) -> None:
        pass

    def saving_input_vars_histograms(self, df, output_path: str, suptitle: str):
        """
        Function to generate and sabe histograms
        """
        df.hist(layout=(10, 4), figsize=(20, 20))
        plt.suptitle(suptitle, fontsize=40)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(output_path)
        plt.show()
        plt.clf()
