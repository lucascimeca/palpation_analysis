from experiment_setup import ExperimentSetup
import numpy as np
import matplotlib.pyplot as plt
import pathlib

SHOW = False
SAVE_LOCAL = False
SAVE_GLOBAL = False
SAVE_FORMAT = 'eps'
RESOLUTION = 1000
environments = ['phantom1', 'phantom2']


if __name__ == "__main__":

    properties_exp_state = {
        'phantom1': {
            '1-1': 'NA', '1-2': 'BS', '1-3': 'NA', '1-4': 'NA',
            '2-1': 'SD', '2-2': 'NA', '2-3': 'NA', '2-4': 'NA',
            '3-1': 'NA', '3-2': 'NA', '3-3': 'SS', '3-4': 'NA',
            '4-1': 'NA', '4-2': 'NA', '4-3': 'NA', '4-4': 'BD'
        },
        'phantom2': {
            '1-1': 'BD', '1-2': 'SS', '1-3': 'BD', '1-4': 'SS',
            '2-1': 'BS', '2-2': 'NA', '2-3': 'SS', '2-4': 'SD',
            '3-1': 'NA', '3-2': 'BD', '3-3': 'NA', '3-4': 'SD',
            '4-1': 'BS', '4-2': 'NA', '4-3': 'BS', '4-4': 'SD'
        }
    }
    # pca_sample_numb_change = {
    #     '{3xNA, 1xSD}': ['3-1', '2-2', '3-3', '3-4'],
    #     '{2xNA, 1xSD, 1xSS}': ['3-1', '2-2', '3-4', '1-4'],
    #     '{1xNA, 1xSS, 1xSD, 1xBD}': ['3-1', '3-4', '1-4', '1-1'],
    #     '{1xNA, 1xSS, 1xBD, 1xBS}': ['3-1', '4-3', '1-3', '4-3'],
    # }
    pca_sample_numb_change = {
        'no BS':    ['1-1', '3-1', '1-2', '2-2', '3-2', '4-2', '1-3', '2-3', '3-3', '1-4', '2-4', '3-4', '4-4'],
        'no BD':    ['2-1', '3-1', '4-1', '1-2', '2-2', '4-2', '2-3', '3-3', '4-3', '1-4', '2-4', '3-4', '4-4'],
        'no SS':    ['1-1', '2-1', '3-1', '4-1', '2-2', '3-2', '4-2', '1-3', '3-3', '4-3', '2-4', '3-4', '4-4'],
        'no SD':    ['1-1', '2-1', '3-1', '4-1', '1-2', '2-2', '3-2', '4-2', '1-3', '2-3', '3-3', '4-3', '1-4'],
    }
    data_path = {
        'phantom1': "./../data/presence_exp.mat",
        'phantom2': "./../data/properties_exp.mat"
    }


    exp_metric_data = dict()
    exp_fig_data = dict()
    exp_setup = dict()

    env = 'phantom2'
    for key in pca_sample_numb_change.keys():
        exp_setup[key] = ExperimentSetup(data_path[env], baseline="./../data/baseline.mat")
        exp_setup[key].set_class_names(properties_exp_state[env])
        exp_metric_data[key], _ = exp_setup[key].run_experiments(
            which_type='presence',
            which_clustering='k_means',
            show=SHOW,
            save_local=SAVE_LOCAL,
            save_global=SAVE_GLOBAL,
            where=env,
            resolution=RESOLUTION,
            format=SAVE_FORMAT,
            # bad extra parameters
            pca_sample_numb_change=pca_sample_numb_change[key]
            )

    # -------------------------------------------------------------------------
    # ---------------------------   FIGURES -----------------------------------
    # -------------------------------------------------------------------------

    # create path for quality figures
    pathlib.Path('../manuscript/generated_figures/quality/').mkdir(parents=True, exist_ok=True)
    markers = ['^', 'v', 'o', '<', '>']

    figures = dict()
    #  -----------------   FIGURE 8 - EXPLAINED VARIANCE ------------------------ #

    # Plot Vertical motion
    vertical_var_fig = plt.figure(figsize=(13, 8))
    ax = vertical_var_fig.add_subplot(111)
    ax.set_xlabel('depth ($mm$)', fontsize=26)
    ax.set_ylabel('explained variance (%)', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for key, i in zip(pca_sample_numb_change.keys(), list(range(len(list(pca_sample_numb_change.keys()))))):
        plt.plot(exp_metric_data[key]['Vertical']['xs'],
                 exp_metric_data[key]['Vertical']['explained_variance_ratios']*100,
                 marker=markers[i],
                markersize=12,
                 linestyle='-',
                 label=key)
    plt.legend(fontsize=18)
    figures['vertical_var_fig'] = {
        'location': '../manuscript/generated_figures/quality/explained_variance_vertical_quality',
        'fig': vertical_var_fig,
        'ax': ax
    }
    plt.close(vertical_var_fig)


    # Plot Rotatory Motion
    rot_var_fig = plt.figure(figsize=(13, 8))
    ax = rot_var_fig.add_subplot(111)
    ax.set_xlabel('radius ($mm$)', fontsize=26)
    ax.set_ylabel('explained variance (%)', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for key, i in zip(pca_sample_numb_change.keys(), list(range(len(list(pca_sample_numb_change.keys()))))):
        which_rad = np.argmax(exp_metric_data[key]['Rotate']['explained_variance']) %\
                    exp_metric_data[key]['Rotate']['explained_variance'].shape[1]

        plt.plot(exp_metric_data[key]['Rotate']['ys'][:, 0],
                 exp_metric_data[key]['Rotate']['explained_variance'][:, which_rad],
                 marker=markers[i],
                markersize=12,
                 linestyle='-',
                 label=key
                       + ', $d$=' + str(exp_metric_data[key]['Rotate']['xs'][0, which_rad]) + '$mm$')
    plt.legend(fontsize=18)
    figures['rot_var_fig'] = {
        'location': '../manuscript/generated_figures/quality/explained_variance_rotation_quality',
        'fig': rot_var_fig,
        'ax': ax
    }
    plt.close(rot_var_fig)

    #  -----------------   FIGURE 9 - SILHOUETTE------------------------ #
    # Plot Vertical motion

    ver_sc_fig = plt.figure(figsize=(13, 8))
    ax = ver_sc_fig.add_subplot(111)
    ax.set_xlabel('depth ($mm$)', fontsize=26)
    ax.set_ylabel('silhouette coefficient', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for key, i in zip(pca_sample_numb_change.keys(), list(range(len(list(pca_sample_numb_change.keys()))))):
        plt.plot(exp_metric_data[key]['Vertical']['xs'],
                 exp_metric_data[key]['Vertical']['silhouette_coefficient'],
                 marker=markers[i],
                markersize=12,
                 linestyle='-',
                 label=key)
    plt.legend(fontsize=18)
    figures['ver_sc_fig'] = {
        'location': '../manuscript/generated_figures/quality/silhouette_coefficient_vertical_quality',
        'fig': ver_sc_fig,
        'ax': ax
    }
    plt.close(ver_sc_fig)

    # Plot Rotatory Motion
    rot_sc_fig = plt.figure(figsize=(13, 8))
    ax = rot_sc_fig.add_subplot(111)
    ax.set_xlabel('radius ($mm$)', fontsize=26)
    ax.set_ylabel('silhouette coefficient', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for key, i in zip(pca_sample_numb_change.keys(), list(range(len(list(pca_sample_numb_change.keys()))))):
        which_rad = np.argmax(exp_metric_data[key]['Rotate']['silhouette_coefficient']) %\
                    exp_metric_data[key]['Rotate']['silhouette_coefficient'].shape[1]

        plt.plot(exp_metric_data[key]['Rotate']['ys'][:, 0],
                 exp_metric_data[key]['Rotate']['silhouette_coefficient'][:, which_rad],
                 marker=markers[i],
                markersize=12,
                 linestyle='-',
                 label=key
                       + ', $d$=' + str(exp_metric_data[key]['Rotate']['xs'][0, which_rad]) + '$mm$')
    plt.legend(fontsize=18)
    figures['rot_sc_fig'] = {
        'location': '../manuscript/generated_figures/quality/silhouette_coefficient_rotation_quality',
        'fig': rot_sc_fig,
        'ax': ax
    }
    plt.close(rot_sc_fig)


    # get plot limits variance
    var_keys = list(np.unique([key for key in figures.keys() if key.split('_')[1] == 'var']))
    ylim = list(figures[var_keys[0]]['ax'].get_ylim())
    for i in range(1, len(var_keys)):
        ylim = [min(figures[var_keys[i]]['ax'].get_ylim()[0], ylim[0]),
                max(figures[var_keys[i]]['ax'].get_ylim()[1], ylim[1])]

    # plot stuff
    for key in var_keys:
        fig = figures[key]['fig']
        ax = figures[key]['ax']
        # ax.set_ylim(ylim)
        fig.tight_layout()
        fig.savefig(
            figures[key]['location'] + '.' + SAVE_FORMAT,
            bbox_inches="tight"
        )

    # get plot limits silhouette coefficient
    sc_keys = list(np.unique([key for key in figures.keys() if key.split('_')[1] == 'sc']))
    ylim = list(figures[sc_keys[0]]['ax'].get_ylim())
    for i in range(1, len(var_keys)):
        ylim = [min(figures[sc_keys[i]]['ax'].get_ylim()[0], ylim[0]),
                max(figures[sc_keys[i]]['ax'].get_ylim()[1], ylim[1])]

    # plot stuff
    for key in sc_keys:
        fig = figures[key]['fig']
        ax = figures[key]['ax']
        # ax.set_ylim(ylim)
        fig.tight_layout()
        fig.savefig(
            figures[key]['location'] + '.' + SAVE_FORMAT,
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
    # colors = ['#ff7575', '#8fe89e', '#7f91e8']
    patterns = ['', '-', '/']
    for key, i in zip(pca_sample_numb_change.keys(), list(range(len(list(pca_sample_numb_change.keys()))))):
        print(key)
        x = exp_metric_data[key]['Vertical']['xs']
        y = exp_metric_data[key]['Vertical']['svm_accuracies']
        sil = exp_metric_data[key]['Vertical']['silhouette_coefficient']
        ax.plot(x, y,
                # color=colors[i],
                ms=10,
                markeredgecolor='k',
                marker='o',
                mec=None,
                alpha=.5,
                linewidth=4,
                label='$' + str(key) + '$' + ' classes')

        ax.scatter([x[np.argmax(sil)]], [y[np.argmax(sil)]],
                   marker='o',
                   c='k',
                   linewidth=3,
                   s=1550,
                   alpha=.7)
        # shift += width
    ax.set_xticks(x)
    ax.set_xticklabels([str(elem) for elem in x])
    plt.legend(fontsize=20)
    clst_no_fig.tight_layout()
    clst_no_fig.savefig(
        '../manuscript/generated_figures/bar_quality_num_change_svm' + '.' + 'eps',
        bbox_inches="tight"
    )
    clst_no_fig.savefig(
        '../manuscript/generated_figures/bar_quality_num_change_svm' + '.' + 'png',
        bbox_inches="tight"
    )