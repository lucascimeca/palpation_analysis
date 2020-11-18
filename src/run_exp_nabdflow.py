from experiment_setup import ExperimentSetup
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

SHOW = False
SAVE_LOCAL = False
SAVE_GLOBAL = True
SAVE_FORMAT = 'eps'
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
    '{2xBS, 2xNA}': ['2-1', '3-1', '4-1', '4-2'],
}
data_path = {
    'phantom2': "./../data/properties_exp.mat"
}


def print_clust_flow(fig, ax, X_init, X_fin, fig_data, show=False):

    # get data from experiment
    # resolution = fig_data['resolution']
    markers = fig_data['markers']
    targets = fig_data['targets']
    # n_clusters = fig_data['n_clusters']
    cls_to_target_dict = fig_data['cls_to_target_dict']
    tactile_sectors = fig_data['tactile_sectors']
    class_names = fig_data['class_names']
    # target_to_cls_dict = fig_data['target_to_cls_dict']
    # ax_cpy = fig_data['ax']

    # k_means_cluster_centers = fig_data['k_means_cluster_centers']
    # tactile_sectors = fig_data['tactile_sectors']
    # class_names = fig_data['class_names']
    target_to_cls_dict = fig_data['target_to_cls_dict']
    # center_class_labels = fig_data['center_class_labels']

    # # set parameters for mash
    # x_max = max(X_init[:, 0].max(), X_fin[:, 0].max())
    # x_min = min(X_init[:, 0].min(), X_fin[:, 0].min())
    # y_max = max(X_init[:, 1].max(), X_fin[:, 1].max())
    # y_min = min(X_init[:, 1].min(), X_fin[:, 1].min())
    # h = np.mean(x_max + 1 - x_min - 1) / resolution
    # x_margin = np.mean((x_max + 1 - x_min - 1) / 500) * 30
    # y_margin = np.mean((x_max + 1 - x_min - 1) / 500) * 30

    # xx_min, xx_max = x_min - x_margin, x_max + x_margin
    # yy_min, yy_max = y_min - y_margin, y_max + y_margin
    # xx, yy = np.meshgrid(np.arange(xx_min, xx_max, h), np.arange(yy_min, yy_max, h))

    # Create color maps
    colors_light = ['#c6f0ff', '#ffb2a8', '#edffa8', '#a8ffc2', '#c2a8ff']
    colors_bold = ['#0191c4', '#b71500', '#d1d800', '#00b235', '#3400af']
    cmap_light = ListedColormap(colors_light)
    cmap_bold = ListedColormap(colors_bold)

    key = list(pca_sample_numb_change.keys())[0]
    types = set()

    # plot data points
    for sect in pca_sample_numb_change[key]:
        types.add(class_names[sect])
        ind = [i for i in range(len(tactile_sectors)) if tactile_sectors[i] == sect][0]
        ax.scatter(
            X_init[ind, 0], X_init[ind, 1],
            c=targets[ind],
            cmap=cmap_light,
            marker=markers[targets[ind]],
            edgecolor='k',
            s=640,
            vmin=0,
            vmax=len(colors_bold)
        )
        ax.scatter(
            X_fin[ind, 0], X_fin[ind, 1],
            c=targets[ind],
            cmap=cmap_bold,
            marker=markers[targets[ind]],
            edgecolor='k',
            s=640,
            vmin=0,
            vmax=len(colors_bold)
        )
        # add motion arrows
        # XQ, YQ = np.meshgrid(np.arange(xx_min, xx_max, 1000), np.arange(yy_min, yy_max, 1000))
        ax.quiver(X_init[ind, 0], X_init[ind, 1], X_fin[ind, 0] - X_init[ind, 0], X_fin[ind, 1] - X_init[ind, 1],
                  angles='xy', scale_units='xy', scale=1, width=0.003, alpha=0.8, headaxislength=4.5, headlength=6.5,
                  headwidth=5.5)

    # Centers
    centers = []
    for type, i in zip(types, range(len(types))):
        sects = [sect for sect in pca_sample_numb_change[key] if class_names[sect]==type]
        ids = []
        for sect in sects:
            ids += [i for i in range(len(tactile_sectors)) if tactile_sectors[i] == sect]
        center = np.average(X_fin[ids, :], axis=0)
        centers += [tuple(center)]
        ax.scatter(
            center[0], center[1],
            c=targets[ids[0]],
            cmap=cmap_bold,
            linewidth=3,
            marker='+',
            s=670,
            edgecolor='k',
            vmin=0,
            vmax=len(colors_bold)
        )
        for j in range(len(ids)):
            plt.plot([center[0], X_fin[ids[j], 0]], [center[1], X_fin[ids[j], 1]], ls='--', color='#fc9211', lw=3)
            if j==0:
                plt.text(np.average([center[0], X_fin[ids[j], 0]]), np.average([center[1], X_fin[ids[j], 1]]),
                     "$a("+str(i)+")$",#+str(j)+"}$"
                     fontsize=26)

    plt.plot([centers[0][0], centers[1][0]], [centers[0][1], centers[1][1]], ls='-.', color='#0461f7', lw=3)
    plt.text(np.average([centers[0][0], centers[1][0]]), np.average([centers[0][1], centers[1][1]])+1000,
             "$b(0)/b(1)$",
             fontsize=26)

    # xlim = list(ax_cpy.get_xlim())
    # ylim = list(ax_cpy.get_ylim())
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    # add legends
    patches = []
    tar_labels = set()
    for i in [cls_to_target_dict[class_names[sect]] for sect in pca_sample_numb_change[key]]:
        if target_to_cls_dict[i] not in tar_labels:
            patches.append(
                plt.plot(
                    [], [],
                    marker=markers[i],
                    ms=20,
                    markeredgecolor='k',
                    ls="",
                    mec=None,
                    color=colors_bold[i],
                    label=target_to_cls_dict[i] + ", $\Theta = \{\stackrel{19.5}{0}\}$")[0]
            )
            tar_labels.add(target_to_cls_dict[i])

    tar_labels = set()
    for i in [cls_to_target_dict[class_names[sect]] for sect in pca_sample_numb_change[key]]:
        if target_to_cls_dict[i] not in tar_labels:
            patches.append(
                plt.plot(
                    [], [],
                    marker=markers[i],
                    ms=20,
                    markeredgecolor='k',
                    ls="",
                    mec=None,
                    color=colors_light[i],
                    label=target_to_cls_dict[i] + ", $\Theta = \{\stackrel{6.5}{0}\}$")[0],
            )
            tar_labels.add(target_to_cls_dict[i])
    ax.legend(handles=patches, fontsize=18, loc='best')

    if show:
        plt.show()

    return fig


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

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax.set_ylabel('$\\vec{p}_2$', fontsize=48)

    X_init = exp_fig_data['Vertical-d'+str(np.min(exp_setup.ver_depths))]['X']
    X_fin = exp_fig_data['Vertical-d'+str(np.max(exp_setup.ver_depths))]['X']
    fig = print_clust_flow(fig, ax, X_init, X_fin, exp_fig_data['Vertical-d'+str(np.max(exp_setup.ver_depths))])
    fig.tight_layout()
    fig.savefig(
        '../manuscript/generated_figures/flow' + '.' + SAVE_FORMAT,
        bbox_inches="tight"
    )