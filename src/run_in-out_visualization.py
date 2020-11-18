from experiment_setup import ExperimentSetup
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

SHOW = False
SAVE_LOCAL = False
SAVE_GLOBAL = True
SAVE_FORMAT = 'jpg'
RESOLUTION = 1000
environments = ['phantom1', 'phantom2']

properties_exp_state = {
    'phantom2': {
        '1-1': 'BD', '1-2': 'SS', '1-3': 'BD', '1-4': 'SS',
        '2-1': 'BS', '2-2': 'NA', '2-3': 'SS', '2-4': 'SD',
        '3-1': 'NA', '3-2': 'BD', '3-3': 'NA', '3-4': 'SD',
        '4-1': 'BS', '4-2': 'NA', '4-3': 'BS', '4-4': 'SD'
    }
}
pca_sample_numb_change = {
        '{1xNA, 1xBS, 1xBD, 1xSS, 1xSD}': ['3-1', '2-1', '3-2', '1-4', '2-4'],
}
data_path = {
    'phantom2': "./../data/properties_exp.mat"
}

if __name__ == "__main__":

    env = 'phantom2'
    key = list(pca_sample_numb_change.keys())[0]

    exp_setup = ExperimentSetup(data_path[env], baseline="./../data/baseline.mat")
    exp_setup.set_class_names(properties_exp_state[env])
    exp_metric_data, exp_fig_data = exp_setup.run_experiments(
        which_type='properties',
        which_clustering='k_means',
        show=SHOW,
        save_local=SAVE_LOCAL,
        save_global=SAVE_GLOBAL,
        where=env,
        resolution=RESOLUTION,
        format=SAVE_FORMAT
        )

    # -------------------------------------------------------------------------
    # ---------------------------   FIGURES -----------------------------------
    # -------------------------------------------------------------------------
    #
    # resolution = exp_fig_data['resolution']
    # markers = exp_fig_data['markers']
    # targets = exp_fig_data['targets']
    # n_clusters = exp_fig_data['n_clusters']
    # cls_to_target_dict = exp_fig_data['cls_to_target_dict']
    tactile_sectors = exp_fig_data['Vertical-d14.5']['tactile_sectors']
    # class_names = exp_fig_data['class_names']
    # target_to_cls_dict = exp_fig_data['target_to_cls_dict']
    # ax_cpy = exp_fig_data['ax']

    gs0 = gridspec.GridSpec(2, 5)
    fig = plt.figure(figsize=(33, 15))

    j = 0
    for sect in pca_sample_numb_change[key]:
        ind = [i for i in range(len(tactile_sectors)) if tactile_sectors[i] == sect][0]
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[j])
        ax = plt.Subplot(fig, gs00[:, :])
        ax.set_title(exp_setup.target_to_cls_dict['properties'][exp_setup.targets['properties'][ind]], fontsize=50)
        ax.set_xlabel('time ($steps$)', fontsize=26)
        ax.set_ylabel('input depth ($mm$)', fontsize=26)

        X = np.arange(6, 91, 7)
        step = (14.5-6.5)/91
        Y = [elem*step+6.5  for elem in X]

        fig.add_subplot(ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(X, Y, linestyle='-')
        j += 1

    xlim = []
    ylim = []
    s = j
    for sect in pca_sample_numb_change[key]:
        ind = [i for i in range(len(tactile_sectors)) if tactile_sectors[i] == sect][0]
        gs10 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[j])
        ax = plt.Subplot(fig, gs10[:, :])

        X = np.arange(6, 91, 7)
        Y = exp_setup.experiment_data['Vertical-d14.5'][ind, X]

        fig.add_subplot(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(X, Y, marker='^', linestyle='-')
        j += 1

        if len(xlim) == 0:
            xlim = list(ax.get_xlim())
            ylim = list(ax.get_ylim())
        else:
            xlim = [min(ax.get_xlim()[0], xlim[0]), max(ax.get_xlim()[1], xlim[1])]
            ylim = [min(ax.get_ylim()[0], ylim[0]), max(ax.get_ylim()[1], ylim[1])]

    for sect in pca_sample_numb_change[key]:
        ind = [i for i in range(len(tactile_sectors)) if tactile_sectors[i] == sect][0]
        gs10 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[s])
        ax = plt.Subplot(fig, gs10[:, :])
        ax.set_xlabel('time ($steps$)', fontsize=26)
        ax.set_ylabel('taxel activation', fontsize=26)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        X = np.arange(6, 91, 7)
        Y = exp_setup.experiment_data['Vertical-d14.5'][ind, X]

        fig.add_subplot(ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(X, Y, marker='^', linestyle='-')
        s += 1

fig.tight_layout()
fig.savefig(
    '../manuscript/generated_figures/in-out' + '.' + SAVE_FORMAT,
    bbox_inches="tight"
)