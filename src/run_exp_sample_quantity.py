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
    pca_sample_numb_change = {
        '2 areas: {1xBS, 1xNA}': ['2-1', '3-1'],
        '4 areas: {1xNA, 1xBS, 1xBD, 1xSS}': ['3-1', '2-1', '3-2', '1-4'],
        '8 areas: {2xNA, 2xBS, 2xBD, 2xSS}': ['3-1', '2-1', '3-2', '1-4',
                                              '2-2', '4-3', '1-3', '1-2'],
        '16 areas: {4xNA, 3xBS, 3xBD, 3xSS, 3xSD}': properties_exp_state['phantom2'].keys()
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
            which_type='properties',
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
    pathlib.Path('../manuscript/generated_figures/quantity/').mkdir(parents=True, exist_ok=True)
    markers = ['^', 'v', 'o', '<', '>']

    figures = dict()
    #  -----------------   FIGURE 8 - EXPLAINED VARIANCE ------------------------ #
    # Plot EXPLAINED VARIANCE for Vertical motion

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
        'location': '../manuscript/generated_figures/quantity/explained_variance_vertical_quantity',
        'fig': vertical_var_fig,
        'ax': ax
    }
    plt.close(vertical_var_fig)


    # Plot EXPLAINED VARIANCE for Rotatory Motion
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
        'location': '../manuscript/generated_figures/quantity/explained_variance_rotation_quantity',
        'fig': rot_var_fig,
        'ax': ax
    }
    plt.close(rot_var_fig)

    #  -----------------   FIGURE 9 - SILHOUETTE------------------------ #
    # Plot EXPLAINED VARIANCE for Vertical motion

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
        'location': '../manuscript/generated_figures/quantity/silhouette_coefficient_vertical_quantity',
        'fig': ver_sc_fig,
        'ax': ax
    }
    plt.close(ver_sc_fig)

    # Plot EXPLAINED VARIANCE for Rotatory Motion
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
        'location': '../manuscript/generated_figures/quantity/silhouette_coefficient_rotation_quantity',
        'fig': rot_sc_fig,
        'ax': ax
    }
    plt.close(rot_sc_fig)

    #
    # # get plot limits variance
    var_keys = list(np.unique([key for key in figures.keys() if key.split('_')[1] == 'var']))
    # ylim = list(figures[var_keys[0]]['ax'].get_ylim())
    # for i in range(1, len(var_keys)):
    #     ylim = [min(figures[var_keys[i]]['ax'].get_ylim()[0], ylim[0]),
    #             max(figures[var_keys[i]]['ax'].get_ylim()[1], ylim[1])]

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

    # # get plot limits silhouette coefficient
    sc_keys = list(np.unique([key for key in figures.keys() if key.split('_')[1] == 'sc']))
    # ylim = list(figures[sc_keys[0]]['ax'].get_ylim())
    # for i in range(1, len(var_keys)):
    #     ylim = [min(figures[sc_keys[i]]['ax'].get_ylim()[0], ylim[0]),
    #             max(figures[sc_keys[i]]['ax'].get_ylim()[1], ylim[1])]

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