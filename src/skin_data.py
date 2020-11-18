import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import cluster
import sklearn.metrics as sm
from itertools import permutations
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import normalize as norm
"""
Class handling the skin data
The data is stored in two forms, 'non_refined_data' and 'data'. 
'non_refined_data' is a NxM matrix where M is the number of data points (taxels + heat sensors) and N are the
time frames where the data was sampled
'data' is a MxTxS matrix where M are the number of modules used for sensing, T is the number of taxels per each module
and S are the number of time samples in the data.
"""
class SkinData:
    modules_number = None
    taxels_number = None
    data = None
    non_refined_data = None
    class_names = None
    baseline = None

    def __init__(self, skin_data=None, baseline=None):
        self.data = dict()
        self.non_refined_data = dict()
        if isinstance(baseline, str):
            self.baseline = np.average(loadmat(baseline)['data'], axis=0)
        if skin_data is not None:
            self.load_data(skin_data)
            self._clean_data()

    def get_data(self, which_object=None):
        if which_object:
            return self.data[which_object]
        return self.data

    def get_non_refined_data(self, which_object=None):
        if which_object:
            return self.data[which_object]
        return self.non_refined_data

    def get_taxel(self, key, taxel_number):
        return [self.data[key][i] for i in range(self.data[key].shape[1]) if i % taxel_number == 0]

    def load_data(self, skin_data):
        # load data
        if isinstance(skin_data, dict):
            self.non_refined_data = skin_data
        elif isinstance(skin_data, str):
            self.non_refined_data = {k : v for k,v in loadmat(skin_data).items()
                                     if isinstance(v, np.ndarray) and len(v.shape) == 3
                                     and len(k.split('_')) > 2}  # this line retains only sectors, remove if you want
        else:
            raise NotImplementedError('invalid DATA passed!')

    def _clean_data(self):
        for key in self.non_refined_data.keys():
            if key.split('_')[0] == 'Vertical':
                depth_data = np.squeeze(np.mean(self.non_refined_data[key][:, 1:, :], axis=2))
                ds = int(depth_data.shape[0] / 3)
                for i in range(ds, depth_data.shape[0], int((depth_data.shape[0]-ds)/10)):
                    self.data[key + '_d'+str(self.non_refined_data[key][i, 0, 0])] = (
                        depth_data[i-ds:i, :]
                    ).T.reshape(1, -1)
            else:
                self.data[key] = (
                    np.squeeze(np.mean(self.non_refined_data[key][:,1:,:], axis=2))
                ).T.reshape(1, -1)


"""Function printing the time snapshot of the skin sensor"""
def print_skin_data(experiment_name, skin_data, targets, target_to_cls_dict, no_taxels = 7, where='',
                    which_type='presence', mode='plot', folder='', how_many='all', format='jpg'):
    if how_many == 'all':
        gs0 = gridspec.GridSpec(4, 4)
        for i in range(skin_data.shape[0]):
            fig = plt.figure(figsize=(33, 15))
            img = np.reshape(skin_data[i, :], (no_taxels, -1)).astype(np.int64)
            img =(img.T - np.average(img, axis=1)).T

            gs00 = gridspec.GridSpecFromSubplotSpec(16, 1, subplot_spec=gs0[i])
            ax = plt.Subplot(fig, gs00[:, :])
            ax.set_title(target_to_cls_dict['properties'][targets['properties'][i]], fontsize=50)
            if mode == 'plot':
                for j in range(img.shape[0]):
                    ax = plt.Subplot(fig, gs00[j, :])
                    if j == 0:
                        ax.set_xlabel('time')
                    else:
                        ax.set_xticklabels([])
                    ax.set_ylabel('t'+str(j))
                    fig.add_subplot(ax)
                    plt.plot(range(1, img.shape[1]+1), img[j, :])
            elif mode == 'fig':
                if experiment_name[0] == 'V':
                    # max_depth = float(experiment_name.split('-')[1][1:])
                    # sample_time = img.shape[1]/max_depth
                    # ax.set_xticks(list(np.arange(0, img.shape[1], 2*sample_time)))
                    # ax.set_xticklabels([str(depth) for depth in np.arange(0.5, max_depth, 2)])
                    ax.set_xlabel('time', fontsize=35)
                elif experiment_name[0] == 'R':
                    # max_rot = float(experiment_name.split('-')[2][1:])/10
                    # sample_time = img.shape[1]/max_rot
                    # ax.set_xticks(list(np.arange(0, img.shape[1], 2*fsample_time)))
                    # ax.set_xticklabels([str(deg) for deg in np.arange(0, max_rot, 2)])
                    ax.set_xlabel('time', fontsize=35)
                ax.set_ylabel('taxels', fontsize=35)
                ax.set_yticks(list(range(no_taxels)))
                ax.set_yticklabels(['t'+str(tax) for tax in range(no_taxels)])
                fig.add_subplot(ax)
                plt.imshow(img, cmap='hot')
    else:
        fig = plt.figure(figsize=(13, 6))
        gs0 = gridspec.GridSpec(1, 2)
        nas = [i for i in range(skin_data.shape[0])
               if target_to_cls_dict['properties'][targets['properties'][i]] == 'NA']
        bds = [i for i in range(skin_data.shape[0])
               if target_to_cls_dict['properties'][targets['properties'][i]] == 'BD']
        if len(bds) != 0 and len(nas) != 0:
            idxs = [nas[0]]+[bds[0]]
        else:
            raise ValueError('need to have at least one instance of `ss` and `na` in the experiments')
        # min_intensity = np.min(skin_data[idxs, :])
        # max_intensity = np.max(skin_data[idxs, :])
        for i in range(len(idxs)):
            img = np.reshape(skin_data[idxs[i], :], (no_taxels, -1)).astype(np.int64)
            img = (img.T - np.average(img, axis=1)).T

            gs00 = gridspec.GridSpecFromSubplotSpec(16, 1, subplot_spec=gs0[i])
            ax = plt.Subplot(fig, gs00[:, :])
            ax.set_title(
                target_to_cls_dict['properties'][targets['properties'][idxs[i]]],
                fontsize=35,
                fontweight='bold'
            )
            if mode == 'plot':
                for j in range(img.shape[0]):
                    ax = plt.Subplot(fig, gs00[j, :])
                    if j == 0:
                        ax.set_xlabel('time')
                    else:
                        ax.set_xticklabels([])
                    ax.set_ylabel('t'+str(j))
                    fig.add_subplot(ax)
                    plt.plot(range(1, img.shape[1]+1), img[j, :])
            elif mode == 'fig':
                if experiment_name[0] == 'V':
                    # UNCOMMENT IF NEED DEPTH ON FIG X AXIS
                    # max_depth = float(experiment_name.split('-')[1][1:])
                    # sample_time = img.shape[1]/max_depth
                    # ax.set_xticks(list(np.arange(0, img.shape[1], 2*sample_time)))
                    # ax.set_xticklabels([str(depth) for depth in np.arange(0.5, max_depth, 2)])
                    ax.set_xlabel('time', fontsize=35)
                elif experiment_name[0] == 'R':
                    # UNCOMMENT IF NEED ROTATION ON FIG X AXIS
                    # max_rot = float(experiment_name.split('-')[2][1:])/10
                    # sample_time = img.shape[1]/max_rot
                    # ax.set_xticks(list(np.arange(0, img.shape[1], 2*sample_time)))
                    # ax.set_xticklabels([str(deg) for deg in np.arange(0, max_rot, 2)])
                    ax.set_xlabel('time', fontsize=35)
                ax.set_ylabel('taxels', fontsize=35)
                ax.set_yticks(list(range(no_taxels)))
                ax.set_yticklabels(['t'+str(tax) for tax in range(no_taxels)], fontsize=15)
                fig.add_subplot(ax)
                plt.imshow(img, cmap='hot')

    fig.tight_layout()
    fig.savefig(
        folder + where + which_type + '_rawdata' + mode[0] + '_' + stringify_nums(experiment_name) + '.' + format,
        bbox_inches="tight"
    )
    plt.close(fig)
    return True


def print_taxel_score(taxel_score, experiment_name='', which_type='properties',
                      format='pdf', folder='', where=''):
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    z = np.zeros(xs.shape)
    z[4:6, 1:3] = taxel_score[4]        #
    z[4:6, 3:5] = taxel_score[5]        # BASED ON
    z[2:4, 0:2] = taxel_score[3]        # ___________________
    z[2:4, 2:4] = taxel_score[6]        # |    T4      T5   |
    z[2:4, 4:6] = taxel_score[0]        # | T3     T6    T0 |
    z[0:2, 1:3] = taxel_score[2]        # |    T2      T1   |
    z[0:2, 3:5] = taxel_score[1]        # |_________________|

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=30, elev=55)

    xs = xs.flatten()
    ys = ys.flatten()
    zs = z.flatten()

    hist_colors = plt.cm.Blues(zs.flatten() / float(zs.max()))
    ax.bar3d(xs, ys, np.zeros(len(zs)), 1, 1, zs, color=hist_colors)
    fig.tight_layout()
    fig.savefig(
        folder + where + which_type + '_tax-histogram' + '_' + experiment_name + '.' + format,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Plot with single bars
    # z = np.zeros((3, 3))
    # z[2, 0] = taxel_score[4]
    # z[2, 1] = taxel_score[5]  # BASED ON
    # z[1, 0] = taxel_score[3]  # ___________________
    # z[1, 1] = taxel_score[6]  # |    T4      T5   |
    # z[1, 2] = taxel_score[0]  # | T3     T6    T0 |
    # z[0, 0] = taxel_score[2]  # |    T2      T1   |
    # z[0, 1] = taxel_score[1]
    # xs = np.array([[0.6, 1.6, 2.6], [0.1, 1.1, 2.1], [0.6, 1.6, 2.6]])
    # ys = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1]])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(azim=30, elev=55)
    # xs = xs.flatten()
    # ys = ys.flatten()
    # zs = z.flatten()
    # hist_colors = plt.cm.Blues(zs.flatten() / float(zs.max()))
    # ax.bar3d(xs, ys, np.zeros(len(zs)), .8, .8, zs, color=hist_colors)
    # fig.tight_layout()
    # fig.savefig(
    #     folder + where + which_type + '_tax-histogram' + '_' + experiment_name + '.' + format,
    #     bbox_inches="tight"
    # )

    return True


"""Function applying Kmeans Clustering to a dataset"""
def apply_KMC(X, tactile_sectors, target_to_cls_dict, targets=None,
              class_names=None, n_clusters=2, resolution=10, markers=None, show=True):

    km = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    data_clusters = km.fit_predict(X[:, :2])

    k_means_cluster_centers = km.cluster_centers_
    center_class_labels = km.predict(k_means_cluster_centers)

    # CLUSTER MATCHING
    if targets is not None:
        # match classes for points
        center_class_labels = reorder_clusters(km.labels_, targets)
        data_clusters = np.choose(km.labels_, center_class_labels).astype(np.int64)

    # GET CLUSTER METRICS
    metrics, metric_figures = get_cluster_metrics(
        X,
        data_clusters,
        targets,
        km=km,
        target_to_cls_dict=target_to_cls_dict,
        cluster_centers=k_means_cluster_centers,
        center_class_labels=center_class_labels,
        show=show
    )

    # Plot of relative distance between clusters
    metric_figures['rel_dist_plot'] = plot_c_rel_distance(
        X=X,
        metrics=metrics,
        center_class_labels=center_class_labels,
        markers=markers,
        targets=targets,
        n_clusters=n_clusters,
        k_means_cluster_centers=k_means_cluster_centers,
        target_to_cls_dict=target_to_cls_dict,
        show=show
    )

    # Plot of guesses and data
    # NOTE: after this 'km' will NOT be holding the cluster info anymore!
    metric_figures['classes_plot'], metric_figures['base_plot_data'] = plot_cluster_data(
        km=km,
        X=X,
        resolution=resolution,
        center_class_labels=center_class_labels,
        markers=markers,
        targets=targets,
        n_clusters=n_clusters,
        k_means_cluster_centers=k_means_cluster_centers,
        tactile_sectors=tactile_sectors,
        class_names=class_names,
        target_to_cls_dict=target_to_cls_dict,
        show=show
    )

    return data_clusters, metrics, metric_figures


"""Function renaming the clusters so to match the targets with the highest accuracy
        Inputs: km = KMeans class from sklearn library
                targets = np array of targets for the data
        Output: np array of outputs (same shape as targets)"""
def reorder_clusters(labs, targets):
    lables = labs.copy()
    perms = list(permutations(np.unique(lables)))
    accuracies = [sm.accuracy_score(targets, np.choose(lables, perms[i]).astype(np.int64))
                  for i in range(len(perms))]
    return perms[np.argmax(accuracies)]


"""Function computing some metrics for the clustering"""
def get_cluster_metrics(X, outputs, targets, km=None, target_to_cls_dict=None,
                        cluster_centers=None, center_class_labels=None, show=True):
    metrics = dict()
    metric_figures = dict()

    n = np.unique(outputs).shape[0]

    # ------ CONFUSION MATRIX -------
    conf_fig = plt.figure(figsize=(10, 5))
    ax = conf_fig.add_subplot(1, 1, 1)
    cm = sm.confusion_matrix(targets, outputs)
    plt.imshow(cm, interpolation='none', cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center', fontsize=24)
    plt.xlabel("kmeans labels", fontsize=16)
    plt.ylabel("true labels", fontsize=16)
    plt.xticks(range(n), [target_to_cls_dict[i] + "\nguess" for i in range(n)], size='medium')
    plt.yticks(range(n), [target_to_cls_dict[i] for i in range(n)], size='small')
    metric_figures['confusion_plot'] = conf_fig


    #todo what?
    # plt.imshow(cm, interpolation='none', cmap='Blues')
    # for (i, j), z in np.ndenumerate(cm):
    #     plt.text(j, i, z, ha='center', va='center', fontsize=24)
    # plt.xlabel("kmeans labels", fontsize=16)
    # plt.ylabel("true labels", fontsize=16)

    metrics['accuracy'] = sm.accuracy_score(targets, outputs)
    metrics['fowlkes_mallows_score'] = sm.fowlkes_mallows_score(targets, outputs)
    metrics['silhouette_coefficient'] = sm.silhouette_score(X, targets)  # np.log(np.sum(silhouette_coefficient))

    if show:
        plt.draw()

    if km is not None:
        # ------ DISTANCE MATRIX -------
        dist_mat_fig = plt.figure(figsize=(10, 5))
        ax = dist_mat_fig.add_subplot(1, 1, 1)
        cm = np.zeros((n, n))
        na_dist = np.zeros(n)
        for i in range(n):
            i_idx = np.where(np.array(center_class_labels) == i)[0][0]  # actual i index after cluster matching
            for j in range(n):
                j_idx = np.where(np.array(center_class_labels) == j)[0][0]  # actual j index after cluster matching
                if i_idx != j_idx:
                    cm[i, j] = np.sqrt(np.dot(
                        (cluster_centers[i_idx, :] - cluster_centers[j_idx, :]),
                        (cluster_centers[i_idx, :] - cluster_centers[j_idx, :]).T
                    ))
                    if target_to_cls_dict[i] == 'NA':
                        na_dist[j] = cm[i, j]
        max_dist = np.max(cm)
        plt.imshow(max_dist-cm, interpolation='none', cmap='Greens')
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, s='%.2f' % (z/max_dist), ha='center', va='center', fontsize=16)
        # plt.xlabel("kmeans labels", fontsize=16)
        # plt.ylabel("true labels", fontsize=16)
        plt.xticks(range(n), [target_to_cls_dict[i] for i in range(n)], size='medium')
        plt.yticks(range(n), [target_to_cls_dict[i] for i in range(n)], size='medium')
        metric_figures['dist_mat'] = dist_mat_fig
        metrics['na_dist'] = na_dist
        plt.close(dist_mat_fig)
    return metrics, metric_figures


"""Function to plot the data from the KMC algorithm"""
def plot_cluster_data(km, X, resolution, center_class_labels, markers, targets, n_clusters,
                      k_means_cluster_centers, tactile_sectors, class_names, target_to_cls_dict, show):

    # Create color maps
    colors_light = ['#c6f0ff', '#ffb2a8', '#edffa8', '#a8ffc2', '#c2a8ff']
    colors_bold = ['#0191c4', '#b71500', '#d1d800', '#00b235', '#3400af']
    cmap_light = ListedColormap(colors_light)
    cmap_bold = ListedColormap(colors_bold)

    # plot mash
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax.set_ylabel('$\\vec{p}_2$', fontsize=48)

    # plot data points
    if markers is None:
        ax.scatter(
            X[:, 0], X[:, 1],
            c=targets,
            cmap=cmap_bold,
            edgecolor='k',
            s=640,
            vmin=0,
            vmax=len(colors_bold),
            alpha=0.7
        )
    else:
        for i in range(n_clusters):
            ids = i == np.array(targets)  # select elements of class i
            ax.scatter(
                X[ids, 0], X[ids, 1],
                c=targets[ids],
                cmap=cmap_bold,
                marker=markers[i],
                edgecolor='k',
                s=640,
                vmin=0,
                vmax=len(colors_bold),
                alpha=0.9
            )

    # draw circles around centroids
    axes = plt.axis()
    for i in range(n_clusters):
        clst_data = X[targets == i, :]

        if clst_data.shape[0] == 1:
            # base ellipse on one point
            clst_center = clst_data[0, :]
            w = ((axes[1] - axes[0]) / 10)
            h = ((axes[3] - axes[2]) / 6.5)
            theta = 0
        else:
            # base ellipse on cov matrix
            clst_center = np.average(clst_data, axis=0)
            clst_cov = np.cov(clst_data, rowvar=False)

            vals, vecs = np.linalg.eigh(clst_cov)  # Compute eigenvalues and associated eigenvectors
            x, y = vecs[:, 0]
            theta = np.degrees(np.arctan2(y, x))  # Compute "tilt" of ellipse using first eigenvector
            w, h = 4 * np.sqrt(vals)  # Eigenvalues give length of ellipse along each eigenvector

        circle = mpatches.Ellipse(clst_center, w, h, theta,
                                  linestyle='--',
                                  color=colors_bold[i],
                                  fill=False,
                                  linewidth=2.5)
        ax.add_artist(circle)
        # draw Centroids
        ax.scatter([clst_center[0]], [clst_center[1]],
            marker='+',
            c='k',
            linewidth=3,
            s=670
        )
        # add center labels
        ax.text(clst_center[0], clst_center[1],
                "$" + target_to_cls_dict[i] + "$",
                style='italic',
                fontsize=34)

    if n_clusters == 2:
        # When two classes plot linear boundary explicitly
        # draw line between centroids
        plt.plot(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1], 'k--', linewidth=1)

        # draw line separating data
        axes = plt.axis()
        y_max = axes[3]
        y_min = axes[2]
        center_x = (k_means_cluster_centers[0, 0]+k_means_cluster_centers[1, 0])/2
        center_y = (k_means_cluster_centers[0, 1]+k_means_cluster_centers[1, 1])/2
        slope = -  (k_means_cluster_centers[1, 0]-k_means_cluster_centers[0, 0])/\
                   (k_means_cluster_centers[1, 1]-k_means_cluster_centers[0, 1])
        shift = center_y - slope * center_x
        p1_y = y_min
        p1_x = (p1_y - shift)/slope
        p2_y = y_max
        p2_x = (p2_y - shift)/slope
        plt.plot([p1_x, p2_x], [p1_y, p2_y], 'k-', linewidth=4)
        plt.xlim([axes[0], axes[1]])
        plt.ylim([axes[2], axes[3]])
        ax.text(center_x, center_y, "$l_{KMC}$", fontsize=34)

        #  ----- ADD PLOT LABELS AND LEGENDS -------
        # plot class name
        if tactile_sectors is not None:
            for i in range(X.shape[0]):
                if targets[i] > 0:
                    mid_x = (max(X[:,0])+min(X[:,0]))/2
                    mid_y = (max(X[:,1])+min(X[:,1]))/2
                    h_align = 'left'
                    v_align = 'bottom'
                    if mid_x-X[i, 0] < 0:
                        h_align = 'right'
                    if mid_y-X[i, 1] < 0:
                        v_align = 'top'
                    if class_names:
                        ax.text(
                            X[i, 0], X[i, 1], class_names[tactile_sectors[i]],
                            horizontalalignment=h_align,
                            verticalalignment=v_align,
                            style='italic',
                            fontsize=32
                        )
                    else:
                        ax.text(
                            X[i,0], X[i, 1], tactile_sectors[i],
                            horizontalalignment=h_align,
                            verticalalignment=v_align,
                            style='italic',
                            fontsize=32
                        )

    # add legends
    cluster_objects = np.sort([target_to_cls_dict[label] for label in center_class_labels])
    patches = []
    cls_to_target_dict = {v: k for k, v in target_to_cls_dict.items()}  # invert dictionary, bad computationally
    for i in range(n_clusters):
        patches.append(
            plt.plot([], [],
                     marker=markers[cls_to_target_dict[cluster_objects[i]]],
                     ms=20,
                     ls="",
                     markeredgecolor='k',
                     mec=None,
                     color=colors_bold[cls_to_target_dict[cluster_objects[i]]],
                     label=target_to_cls_dict[cls_to_target_dict[cluster_objects[i]]])[0]
        )
    ax.legend(handles=patches, fontsize=18, loc='best')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if show:
        plt.show()

    basic_plot_data = {
        'X': X,
        'resolution': resolution,
        'center_class_labels': center_class_labels,
        'markers': markers,
        'targets': targets,
        'n_clusters': n_clusters,
        'k_means_cluster_centers': k_means_cluster_centers,
        'tactile_sectors': tactile_sectors,
        'class_names': class_names,
        'cluster_objects': cluster_objects,
        'target_to_cls_dict': target_to_cls_dict,
        'cls_to_target_dict': cls_to_target_dict,
        'ax': ax,
    }

    return fig, basic_plot_data


def plot_c_rel_distance(X, center_class_labels, targets, k_means_cluster_centers, n_clusters,
                        markers=None, target_to_cls_dict=None, metrics=None, show=False):
    # Create color maps
    colors_bold = ['#0191c4', '#b71500', '#d1d800', '#00b235', '#3400af']
    cmap_bold = ListedColormap(colors_bold)

    # plot initialization
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax.set_ylabel('$\\vec{p}_2$', fontsize=48)

    # plot data points
    if markers is None:
        ax.scatter(
            X[:, 0], X[:, 1],
            c=targets,
            cmap=cmap_bold,
            edgecolor='k',
            s=640,
            vmin=0,
            vmax=len(colors_bold),
                alpha=0.6
        )
    else:
        for i in range(n_clusters):
            ids = i == np.array(targets)  # select elements of class i
            ax.scatter(
                X[ids, 0], X[ids, 1],
                c=targets[ids],
                cmap=cmap_bold,
                marker=markers[i],
                edgecolor='k',
                s=640,
                vmin=0,
                vmax=len(colors_bold),
                alpha=0.6
            )

    # draw circles around centroids
    axes = plt.axis()
    width = (axes[1] - axes[0]) / 6
    height = (axes[3] - axes[2]) / 3.8
    for i in range(n_clusters):
        circle = mpatches.Ellipse(k_means_cluster_centers[i, :], width, height,
                                  linestyle='--',
                                  color=colors_bold[center_class_labels[i]],
                                  fill=False,
                                  linewidth=2.5)
        ax.add_artist(circle)
        ax.text(k_means_cluster_centers[i, 0], k_means_cluster_centers[i, 1],
                "$" + target_to_cls_dict[center_class_labels[i]] + "$",
                style='italic',
                fontsize=34)

    # draw line between centroids
    na_dist_idxs = np.argsort(metrics['na_dist'])
    which_center = np.argsort(center_class_labels)
    idx_from = which_center[na_dist_idxs[0]]
    for i in na_dist_idxs[1:]:
        plt.plot([k_means_cluster_centers[idx_from, 0], k_means_cluster_centers[which_center[i], 0]],
                 [k_means_cluster_centers[idx_from, 1], k_means_cluster_centers[which_center[i], 1]],
                 'k-', linewidth=3, alpha=0.8)
        idx_from = which_center[i]

    # draw Centroids
    ax.scatter(
        k_means_cluster_centers[:, 0],
        k_means_cluster_centers[:, 1],
        marker='+',
        c='k',
        linewidth=3,
        s=250,
        alpha=1
    )

    # add legends
    cluster_objects = [target_to_cls_dict[label] for label in center_class_labels]
    patches = []
    cls_to_target_dict = {v: k for k, v in target_to_cls_dict.items()}  # invert dictionary, bad computationally
    for i in range(n_clusters):
        patches.append(
            plt.plot([], [],
                     marker=markers[cls_to_target_dict[cluster_objects[i]]],
                     ms=20,
                     ls="",
                     markeredgecolor='k',
                     mec=None,
                    alpha=0.6,
                     color=colors_bold[cls_to_target_dict[cluster_objects[i]]],
                     label=target_to_cls_dict[cls_to_target_dict[cluster_objects[i]]] + "$_{true}$")[0]
        )
    ax.legend(handles=patches, fontsize=18, loc='best')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if show:
        plt.show()

    return fig

def stringfy_set(set):
    out_str = "{ "
    for elem in set:
        out_str += str(elem) + ', '
    return out_str[:-2] + " }"


def stringify_nums(name):
    return ''.join(['_' if char == '.' else char for char in name])