import os
import matplotlib as mpl
if os.environ.get('DISPLAY') is None:  # NOQA
    mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
import numpy as np


# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model  # NOQA
def plot_attention(
        input_labels, output_labels, attentions,
        out_colors=None, filepath=None):
    from pylab import rcParams
    rcParams['figure.figsize'] = 16, 16
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xticklabels([''] + output_labels, rotation=90)
    ax.set_yticklabels([''] + input_labels)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if out_colors:
        out_colors = ["k"] + out_colors
        assert len(out_colors) == 1 + len(output_labels), \
            f"got {len(out_colors)} colors for {len(output_labels)} labels"
        for xtick, color in zip(ax.get_xticklabels(), out_colors):
            xtick.set_color(color)

    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html  # NOQA
def plot_confusion_matrix(
        cm, classes,
        normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
        filepath=None):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from pylab import rcParams
    rcParams['figure.figsize'] = 16, 16
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(
            cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
                j, i, f"{cm[i, j]:.2f}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    plot_attention(
        "1 2 3 4".split(),
        "a b c d".split(),
        np.random.rand(4, 4),
        out_colors="r g b r".split())
