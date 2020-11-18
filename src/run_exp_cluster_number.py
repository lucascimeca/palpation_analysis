from experiment_setup import ExperimentSetup
import matplotlib.pyplot as plt
import numpy as np

SHOW = False
SAVE_LOCAL = False
SAVE_GLOBAL = False
RESOLUTION = 1000
SAVE_FORMAT = 'eps'
environments = ['phantom2']

properties_exp_state = {
    'phantom2': {
        '1-1': 'BD', '1-2': 'SS', '1-3': 'BD', '1-4': 'SS',
        '2-1': 'BS', '2-2': 'NA', '2-3': 'SS', '2-4': 'SD',
        '3-1': 'NA', '3-2': 'BD', '3-3': 'NA', '3-4': 'SD',
        '4-1': 'BS', '4-2': 'NA', '4-3': 'BS', '4-4': 'SD'
    }
}
data_path = {
    'phantom2': "./../data/properties_exp.mat"
}

class_types = {
    2: {
        'properties': {
            'BS': 1,
            'BD': 1,
            'SS': 1,
            'SD': 1,
            'NA': 0
        }
    },
    3: {
        'properties': {
            'BS': 2,
            'BD': 2,
            'SS': 1,
            'SD': 1,
            'NA': 0
        }
    },
    5: {
        'properties': {
            'BS': 4,
            'BD': 3,
            'SS': 2,
            'SD': 1,
            'NA': 0
        }
    }
}


if __name__ == "__main__":

    cluster_numbers = list(class_types.keys())

    exp_metric_data = dict()
    exp_fig_data = dict()
    exp_setup = dict()

    env = list(properties_exp_state.keys())[0]
    data_path = data_path[list(data_path.keys())[0]]

    for clst_no in cluster_numbers:

        exp_setup[clst_no] = ExperimentSetup(data_path, baseline="./../data/baseline.mat")
        exp_setup[clst_no].set_class_names(properties_exp_state[env])
        exp_setup[clst_no].set_class_type(class_types[clst_no])
        exp_metric_data[clst_no], _ = exp_setup[clst_no].run_experiments(
            which_type='properties',
            which_clustering='k_means',
            show=SHOW,
            save_local=SAVE_LOCAL,
            save_global=SAVE_GLOBAL,
            where=env,
            resolution=RESOLUTION,
            format=SAVE_FORMAT
            )


    ############# BAR PLOT FIGURE SILHOUETTE ##################

    clst_no_fig = plt.figure(figsize=(18, 10))
    width = 0.25
    ax = clst_no_fig.add_subplot(111)
    ax.set_xlabel('depth ($mm$)', fontsize=32)
    ax.set_ylabel('silhouette coefficient', fontsize=32)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    shift = -width
    x = []
    colors = ['#ff7575', '#8fe89e', '#7f91e8']
    patterns = ['', '-', '/']
    for clst_no, i in zip(cluster_numbers, range(len(cluster_numbers))):
        print(clst_no)
        x = exp_metric_data[clst_no]['Vertical']['xs']
        y = exp_metric_data[clst_no]['Vertical']['silhouette_coefficient']
        ax.bar(x + shift, y,
               width=width,
               color=colors[i],
               edgecolor=['k'] * len(x),
               align='center',
               hatch=patterns[i],
               label='$' + str(clst_no) + '$' + ' clusters')
        shift += width
    ax.set_xticks(x)
    ax.set_xticklabels([str(elem) for elem in x])
    plt.legend(fontsize=20)

    clst_no_fig.tight_layout()
    clst_no_fig.savefig(
            '../manuscript/generated_figures/bar_clst_num_change' + '.' + SAVE_FORMAT,
            bbox_inches="tight"
    )

    ############# BAR PLOT FIGURE SVM ACCURACIES ##################

    clst_no_fig = plt.figure(figsize=(18, 10))
    # width = 0.25
    ax = clst_no_fig.add_subplot(111)
    ax.set_xlabel('depth ($mm$)', fontsize=32)
    ax.set_ylabel('accuracy', fontsize=32)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # shift = -width
    x = []
    colors = ['#ff7575', '#8fe89e', '#7f91e8']
    patterns = ['', '-', '/']
    for clst_no, i in zip(cluster_numbers, range(len(cluster_numbers))):
        print(clst_no)
        x = exp_metric_data[clst_no]['Vertical']['xs']
        y = exp_metric_data[clst_no]['Vertical']['svm_accuracies']
        sil = exp_metric_data[clst_no]['Vertical']['silhouette_coefficient']
        ax.plot(x, y,
                color=colors[i],
                ms=10,
                markeredgecolor='k',
                marker='o',
                mec=None,
                linewidth=4,
                label='$' + str(clst_no) + '$' + ' classes')

        ax.scatter([x[np.argmax(sil)]], [y[np.argmax(sil)]],
                   marker='o',
                   c='k',
                   linewidth=3,
                   s=1550,
                   alpha=.7)

        print("for {} clusters == > min accuracy: {}, avg accuracy {}, max accuracy: {}".format(clst_no, np.min(y), np.average(y), np.max(y)))
        # shift += width
    ax.set_xticks(x)
    ax.set_xticklabels([str(elem) for elem in x])
    plt.legend(fontsize=20)
    clst_no_fig.tight_layout()
    clst_no_fig.savefig(
        '../manuscript/generated_figures/bar_clst_num_change_svm' + '.' + SAVE_FORMAT,
        bbox_inches="tight"
    )