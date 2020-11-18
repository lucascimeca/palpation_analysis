from skin_data import (
    SkinData,
    apply_KMC,
    print_taxel_score,
    print_skin_data,
    stringify_nums
)
import pathlib
import numpy as np
from sklearn import decomposition, svm
import matplotlib.pyplot as plt
from matplotlib import cm as c_map
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import itertools

"""
Class implementing the experiments for --PAPER--. 
experiment_data is a dictionary containing as key the key of the different experiments (eg 3mil, 6mil etc) and data
a tuple containing the data np array as first element and the targets np array as the second.

The matrices names need to be in the format 'classname_experimentname' eg. 'Cube_10mil' for class cube and experiment
10mil
"""
class ExperimentSetup:

    skin = None
    sectors = None
    experiments = None
    trials = set()
    experiment_data = dict()
    pca = None
    targets = dict()
    folder = ''
    ver_depths = None
    rot_rads = None
    rot_depths = None

    class_names = {
        '1-1': 'NA',
        '1-2': 'BS',
        '1-3': 'NA',
        '1-4': 'NA',
        '2-1': 'SD',
        '2-2': 'NA',
        '2-3': 'NA',
        '2-4': 'NA',
        '3-1': 'NA',
        '3-2': 'NA',
        '3-3': 'SS',
        '3-4': 'NA',
        '4-1': 'NA',
        '4-2': 'NA',
        '4-3': 'NA',
        '4-4': 'BD',
    }

    class_type = {
        'presence': {
            'BD': 1,
            'BS': 1,
            'SD': 1,
            'SS': 1,
            'NA': 0
        },
        'properties': {
            'BS': 4,
            'BD': 3,
            'SS': 2,
            'SD': 1,
            'NA': 0
        }
    }

    target_to_cls_dict = {
        'presence': {
            1: "nodule pres",
            0: "nodule abs"
        },
        'properties': {
            4: 'BS',
            3: 'BD',
            2: 'SS',
            1: 'SD',
            0: 'NA'
        }
    }

    markers = {
        'presence': {
            4: '^',
            3: '^',
            2: '^',
            1: '^',
            0: 'o'
        },
        'properties': {
            4: 'p',
            3: 'H',
            2: 'D',
            1: 's',
            0: 'o'
        }
    }

    param_descriptions = {
        'r': 'radius (mm)',
        'd': 'depth (mm)'
    }

    def __init__(self, path_to_data="./../data/skin_experiments.mat", baseline="./../data/baseline.mat"):

        self.skin = SkinData(path_to_data, baseline=baseline)
        all_skin_data = self.skin.get_data()
        self.pca = decomposition.PCA(n_components=2)
        # get classes, sectors etc off of loaded matrices' names
        self.experiments = np.unique([
            '-'.join(exp_name.split('_')[:1]+exp_name.split('_')[3:])
            for exp_name in all_skin_data.keys()
        ])
        self.sectors = np.sort(np.unique(['-'.join(obj.split('_')[1:3]) for obj in all_skin_data.keys()]))
        # load data in the right format for every experiment
        for exp in self.experiments:
            exp_first_key = [key for key in all_skin_data.keys()
                             if exp == '-'.join(key.split('_')[:1] + key.split('_')[3:])
                             and self.sectors[0] == '-'.join(key.split('_')[1:3])][0]

            self.experiment_data[exp] = all_skin_data[exp_first_key]
            for i in range(1, self.sectors.shape[0]):
                exp_key = [key for key in all_skin_data.keys()
                           if exp == '-'.join(key.split('_')[:1] + key.split('_')[3:])
                           and self.sectors[i] == '-'.join(key.split('_')[1:3])][0]
                self.experiment_data[exp] = np.concatenate([
                    self.experiment_data[exp],
                    all_skin_data[exp_key]
                ], axis=0)
        # load targets
        for key in self.class_type.keys():
            self.targets[key] = self._get_targets(which=key)

    def run_experiments(self, which_type='presence', which_clustering='k_means',
                        show=True, save_local=False, save_global=False, resolution=10, where='', format="pdf",
                        pca_sample_numb_change=None):
        if save_local:
            # decide where to put the saved files
            if where == '':
                self.folder = '../manuscript/generated_figures/'
            else:
                self.folder = '../manuscript/generated_figures/' + where + ('/' if where != '' else '')
            pathlib.Path(self.folder).mkdir(parents=True, exist_ok=True)

        # begin experiments
        metrics = dict()
        fig_data = dict()
        for experiment in self.experiments:
            met, exp_figs = self.run_experiment(
                experiment_name=experiment,
                which_clustering=which_clustering,
                which_exp_type=which_type,
                show=show,
                save=save_local,
                where=where,
                resolution=resolution,
                format=format,
                # bas extra parameters
                pca_sample_numb_change=pca_sample_numb_change
            )
            metrics[experiment] = met
            fig_data[experiment] = exp_figs['base_plot_data']
        exp_plot_data = self.__plot_exp_metrics(metrics, which_type, format, where=where, save=save_local)
        plt.close('all')
        return exp_plot_data, fig_data

    def run_experiment(self, experiment_name, which_clustering='k_means', where='',
                       which_exp_type='presence', show=True, save=False,
                       resolution=10, no_taxels=7, format="pdf",
                       pca_sample_numb_change=None):
        # ----  PROJECTION ----
        if pca_sample_numb_change is None:
            self.pca.fit(self.experiment_data[experiment_name])
        else:
            # base pca only on sectors specified in 'pca_sample_numb_change'
            self.pca.fit(self.experiment_data[experiment_name][
                         [True
                          if sect in pca_sample_numb_change
                          else False
                          for sect in self.sectors], :])
        taxel_score = np.sum(
            (self.pca.components_[0, :] / (np.sum(self.pca.components_, axis=1)[0])).reshape(no_taxels, -1),
            axis=1
        )
        projected_data = self.pca.transform(self.experiment_data[experiment_name])
        # ----  CLUSTERING ----
        if which_clustering == 'k_means':
            output_targets, metrics, km_figures = apply_KMC(
                projected_data,
                tactile_sectors=self.sectors,
                target_to_cls_dict=self.target_to_cls_dict[which_exp_type],
                targets=self.targets[which_exp_type],
                class_names=self.class_names,
                resolution=resolution,
                n_clusters=len(np.unique(list({v: k for k, v in self.class_type[which_exp_type].items()}.keys()))),
                markers=self.markers[which_exp_type],
                show=show
            )
        else:
            raise NotImplementedError("Refactored only KMC!")

        # update na_dist matrix

        # test approach -- pick the best vertical motion
        # if experiment_name == 'Vertical-d9.5' or experiment_name == 'Vertical-d14.5':



        # -----------------------------------------------
        # ----------- COMPUTE SVM TEST ------------------

        TRAIN_NO = 1
        CLASS_NO = np.unique(output_targets).shape[0]
        X = projected_data
        y = self.targets[which_exp_type]

        X_train = np.zeros((TRAIN_NO*CLASS_NO, 2))
        X_test = np.zeros((X.shape[0]-X_train.shape[0], 2))
        y_train = np.zeros(X_train.shape[0])
        y_test = np.zeros(X_test.shape[0])

        # split data in training and test set
        idx_train = 0
        idx_test = 0
        for i in range(CLASS_NO):
            idx_train_next = idx_train + TRAIN_NO
            idx_test_next = idx_test + np.sum(y == i) - TRAIN_NO
            X_train[idx_train:idx_train_next, :] = X[y == i, :][:TRAIN_NO, :]
            X_test[idx_test:idx_test_next, :] = X[y == i, :][TRAIN_NO:, :]
            y_train[idx_train:idx_train_next] = i
            y_test[idx_test:idx_test_next] = i
            idx_train = idx_train_next
            idx_test = idx_test_next

        clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #     decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
        #     max_iter=-1, probability=False, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)
        SVM_prediction = clf.predict(X_test)
        SVM_acc = np.sum((SVM_prediction == y_test).astype(np.int8))/SVM_prediction.shape[0]
        # print("{}, Accuracy: {}".format(experiment_name, SVM_acc))

        metrics['svm_accuracy'] = SVM_acc

        # -----------------------------------------------









        # UNCOMMENT IF WANT HISTOGRAM OF EXAMPLE WITH NO-GOOD DATA
        # tmp = np.sum(np.var(projected_data, axis=0)) / \
        #     np.sum(np.var(self.experiment_data[experiment_name], axis=0))
        # if pca_sample_numb_change is not None and tmp > 0.60 and tmp < 0.75\
        #         and len(pca_sample_numb_change) > 4:
        #     pca = decomposition.PCA(n_components=self.experiment_data[experiment_name].shape[1])
        #     pca.fit(self.experiment_data[experiment_name][
        #                  [True
        #                   if sect in pca_sample_numb_change
        #                   else False
        #                   for sect in self.sectors], :])
        #     ps = np.sort(np.var(pca.transform(self.experiment_data[experiment_name]), axis=0) /\
        #         np.sum(np.var(self.experiment_data[experiment_name], axis=0)))
        #     ps = np.concatenate((ps[::-1], np.zeros(self.experiment_data[experiment_name].shape[0]-len(ps))), axis=0)
        #     pca_fig = histogram_plot(
        #         ps,
        #         xlabel='$principal\ components$',
        #         ylabel='$explained\ variance (\\%)$',
        #         xtick_prefix='p'
        #     )
        #     pca_fig.tight_layout()
        #     pca_fig.savefig(
        #         '../manuscript/generated_figures/' + where + which_exp_type + ''.join(pca_sample_numb_change) +
        #         '_pcafig_' + stringify_nums(experiment_name) + '.' + format,
        #         bbox_inches="tight"
        #     )

        if save:
            # saving plots

            # histogram plot(s): good example
            # if pca_sample_numb_change==None:
            #     if experiment_name == 'Vertical-d14.5':
            #         pca = decomposition.PCA(n_components=self.experiment_data[experiment_name].shape[1])
            #         pca.fit(self.experiment_data[experiment_name])
            #         pca_fig = histogram_plot(
            #             pca.explained_variance_ratio_,
            #             xlabel='$principal\ components$',
            #             ylabel='$explained\ variance (\\%)$',
            #             xtick_prefix='p'
            #         )
            #         pca_fig.tight_layout()
            #         pca_fig.savefig(
            #             self.folder + where + which_exp_type + '_pcafig_' + stringify_nums(experiment_name) + '.' + format,
            #             bbox_inches="tight"
            #         )
            #         # plots of na and bd
            #         nabdfig = plotbdbs(sectors=['2-2', '3-2'], data=km_figures['base_plot_data'])
            #         nabdfig.savefig(
            #             self.folder + where + which_exp_type + '_nabdfig_' + stringify_nums(experiment_name) + '.' + format,
            #             bbox_inches="tight"
            #         )


            # confusion matrix plot for experiment
            cm_fig = km_figures['confusion_plot']
            cm_fig.tight_layout()
            cm_fig.savefig(
                self.folder + where + which_exp_type + '_cm_' + stringify_nums(experiment_name) + '.' + format,
                bbox_inches="tight"
            )
            # KMemeans mesh plot for experiment
            cls_plt = km_figures['classes_plot']
            cls_plt.tight_layout()
            cls_plt.savefig(
                self.folder + where + which_exp_type + '_clsplt_' + stringify_nums(experiment_name) + '.' + format,
                bbox_inches="tight"
            )
            # KMemeans mesh plot for experiment
            cls_plt = km_figures['dist_mat']
            cls_plt.tight_layout()
            cls_plt.savefig(
                self.folder + where + which_exp_type + '_distmat_' + stringify_nums(experiment_name) + '.' + format,
                bbox_inches="tight"
            )
            # Cluster distance figure
            cls_plt = km_figures['rel_dist_plot']
            cls_plt.tight_layout()
            cls_plt.savefig(
                self.folder + where + which_exp_type + '_distfig_' + stringify_nums(experiment_name) + '.' + format,
                bbox_inches="tight"
            )
            # Sensor raw data plot over time for experiment
            print_skin_data(
                experiment_name=experiment_name,
                which_type=which_exp_type,
                skin_data=self.experiment_data[experiment_name],
                targets=self.targets,
                target_to_cls_dict=self.target_to_cls_dict,
                format=format,
                folder=self.folder,
                how_many='two',
                where=where,
                mode='fig',
            )
            # plot of taxel score (importance) for experiment given by PCA
            print_taxel_score(
                experiment_name=experiment_name,
                which_type=which_exp_type,
                taxel_score=taxel_score,
                format=format,
                where=where,
                folder=self.folder
            )
            plt.close('all')

        if pca_sample_numb_change is None:
            metrics['explained_variance_ratio'] = np.sum(self.pca.explained_variance_ratio_)
        else:
            metrics['explained_variance_ratio'] = np.sum(np.var(projected_data, axis=0)) /\
                                                  np.sum(np.var(self.experiment_data[experiment_name], axis=0))
        return metrics, km_figures

    def _get_targets(self, which='presence'):
        targets = [self.class_type[which][
                 self.class_names[
                     self.sectors[i]
                 ]] for i in range(len(self.sectors))]
        return np.array(targets)

    def __plot_exp_metrics(self, metrics_dict, which_type, format, where='', save=False):
        exp_plot_data = dict()
        exp_names = np.unique([exp.split('-')[0] for exp in metrics_dict.keys()])
        for exp_name in exp_names:
            exp_plot_data[exp_name] = dict()
            exp_params = np.unique([x[0]
                                    for x in list(itertools.chain(*[exp.split('-')[1:]
                                                                    for exp in metrics_dict.keys()
                                                                    if exp.split('-')[0] == exp_name]))])
            params = []
            for i in range(len(exp_params)):
                if exp_params[i] in self.param_descriptions.keys():
                    params.append(self.param_descriptions[exp_params[i]])
            # get X values for accuracy plot (depths or angles)
            xs_str = np.unique([exp.split('-')[1][1:] for exp in metrics_dict.keys()
                                if exp.split('-')[0] == exp_name])

            if len(exp_params) == 1:
                xs = np.sort([np.int64(np.float(xs_str[i]))if all([d.isdigit() for d in xs_str[i]])
                              else np.float64(xs_str[i]) for i in range(len(xs_str))])
                # get ACCURACY
                accuracies = np.array([
                    metrics_dict[exp_name + '-' + exp_params[0] + str(xs[i])]['accuracy']
                    for i in range(len(xs))
                ])
                # get SILHOUETTE COEFFICIENT
                silhouette_coefficients = np.array([
                    metrics_dict[exp_name + '-' + exp_params[0] + str(xs[i])]['silhouette_coefficient']
                    for i in range(len(xs))
                ])

                svm_accs = np.array([
                    metrics_dict[exp_name + '-' + exp_params[0] + str(xs[i])]['svm_accuracy']
                    for i in range(len(xs))])

                # get EXPLAINED VARIANCE RATIO
                explained_variance_ratios = np.array([
                    metrics_dict[exp_name + '-' + exp_params[0] + str(xs[i])]['explained_variance_ratio']
                    for i in range(len(xs))
                ])

                if len(list(np.unique(self.targets[which_type]))) == 5:
                    # get NA DIST
                    exp_classes = np.sort(list(self.class_type[which_type].keys()))
                    na_dists = np.zeros((len(exp_classes), len(xs)))
                    for i in range(len(xs)) :
                        na_dists[:, i] = metrics_dict[exp_name + '-' + exp_params[0] + str(xs[i])]['na_dist']

                self.ver_depths = xs
                exp_plot_data[exp_name]['xs'] = xs
                exp_plot_data[exp_name]['accuracies'] = accuracies
                exp_plot_data[exp_name]['silhouette_coefficient'] = silhouette_coefficients
                exp_plot_data[exp_name]['explained_variance_ratios'] = explained_variance_ratios
                exp_plot_data[exp_name]['svm_accuracies'] = svm_accs
                # exp_plot_data[exp_name]['na_dists'] = na_dists

                if save:
                    accuracy_fig = plt.figure(figsize=(13, 8))
                    ax = accuracy_fig.add_subplot(111)
                    ax.set_xlabel(params[0], fontsize=16)
                    ax.set_ylabel('accuracy', fontsize=16)
                    plt.plot(xs, accuracies, '-')
                    accuracy_fig.tight_layout()
                    accuracy_fig.savefig(
                        self.folder + where + which_type + exp_name + '_accuracy_' + '.' + format,
                        bbox_inches="tight"
                    )

                    sil_fig = plt.figure(figsize=(13, 8))
                    ax = sil_fig.add_subplot(111)
                    ax.set_xlabel(params[0], fontsize=16)
                    ax.set_ylabel('silhouette coefficient', fontsize=16)
                    plt.plot(xs, silhouette_coefficients, '-')
                    sil_fig.tight_layout()
                    sil_fig.savefig(
                        self.folder + where + which_type + exp_name + '_silhouette' + '.' + format,
                        bbox_inches="tight"
                    )

                    exp_var_fig = plt.figure(figsize=(13, 8))
                    ax = exp_var_fig.add_subplot(111)
                    ax.set_xlabel(params[0], fontsize=16)
                    ax.set_ylabel('explained variance (%)', fontsize=16)
                    plt.plot(xs, explained_variance_ratios*100, '-')
                    exp_var_fig.tight_layout()
                    exp_var_fig.savefig(
                        self.folder + where + which_type + exp_name + '_exp-variance' + '.' + format,
                        bbox_inches="tight"
                    )

                    if len(list(np.unique(self.targets[which_type]))) == 5:
                        na_dist_fig = plt.figure(figsize=(13, 8))
                        ax = na_dist_fig.add_subplot(111)
                        ax.set_xlabel(params[0], fontsize=26)
                        ax.set_ylabel('distance from NA cluster', fontsize=26)
                        plt.xticks(fontsize=16)
                        plt.yticks(fontsize=16)
                        markers = ['^', 'v', 'o', '<', '>']
                        for i in range(na_dists.shape[0]):
                            plt.plot(xs, na_dists[i, :].T,
                                     marker=markers[i],
                                     linestyle='-',
                                    markersize=12,
                                     label=self.target_to_cls_dict[which_type][i])
                        plt.legend()
                        na_dist_fig.tight_layout()
                        na_dist_fig.savefig(
                            self.folder + where + which_type + exp_name + '_na-dist' + '.' + format,
                            bbox_inches="tight"
                        )

            elif len(exp_params) == 2:
                xs = np.sort([np.int64(np.float(xs_str[i]) /10) if all([d.isdigit() for d in xs_str[i]])
                              else np.float64(xs_str[i]) /10 for i in range(len(xs_str))])
                ys_str = np.unique([exp.split('-')[2][1:] for exp in metrics_dict.keys()
                                    if exp.split('-')[0] == exp_name])
                ys = np.sort([np.int64(np.float(ys_str[i])/10) if all([d.isdigit() for d in ys_str[i]])
                              else np.float64(ys_str[i])/10 for i in range(len(ys_str))])

                self.rot_depths = xs
                self.rot_rads = ys

                xs, ys = np.meshgrid(xs, ys)

                def get_z(x, y, which):
                    return metrics_dict[exp_name + '-' +
                                        exp_params[0] + str(x) + '-' +
                                        exp_params[1] + str(y)][which]

                z_acc = np.zeros(xs.shape)
                z_sc = np.zeros(xs.shape)
                z_evs = np.zeros(xs.shape)

                for i in range(len(xs)):
                    for j in range(len(ys)):
                        z_acc[i, j] = get_z(xs[i, j]*10, ys[i, j]*10, 'accuracy')
                        z_sc[i, j] = get_z(xs[i, j]*10, ys[i, j]*10, 'silhouette_coefficient')
                        z_evs[i, j] = get_z(xs[i, j]*10, ys[i, j]*10, 'explained_variance_ratio')*100

                exp_plot_data[exp_name]['xs'] = xs
                exp_plot_data[exp_name]['ys'] = ys
                exp_plot_data[exp_name]['accuracy'] = z_acc
                exp_plot_data[exp_name]['silhouette_coefficient'] = z_sc
                exp_plot_data[exp_name]['explained_variance'] = z_evs

                if save:
                    # ------------- ACCURACY PLOT --------------------
                    # create Accuracy plot figure
                    accuracy_fig = plot_3d_params(
                        xs, ys, z_acc,
                        label_descriptions=params+['accuracy'],
                        which_type=which_type,
                        color_map=c_map.Blues
                    )
                    # save
                    accuracy_fig.tight_layout()
                    accuracy_fig.savefig(
                        self.folder + where + which_type + exp_name + '_accuracy' + '.' + format,
                        bbox_inches="tight",
                    )

                    # --------------- SILHOUETTE PLOT -------------------
                    # create silhuette plot figure
                    accuracy_fig = plot_3d_params(
                        xs, ys, z_sc,
                        label_descriptions=params+['silhouette score'],
                        which_type=which_type,
                        color_map=c_map.Blues
                    )
                    # save
                    accuracy_fig.tight_layout()
                    accuracy_fig.savefig(
                        self.folder + where + which_type + exp_name + '_silhouette' + '.' + format,
                        bbox_inches="tight"
                    )

                    # --------------- EXPLAINED VARIANCE PLOT -------------------
                    # create explained variance plot figure
                    accuracy_fig = plot_3d_params(
                        xs, ys, z_evs,
                        label_descriptions=params+['explained variance (%)'],
                        which_type=which_type,
                        color_map=c_map.Blues
                    )
                    # save
                    accuracy_fig.tight_layout()
                    accuracy_fig.savefig(
                        self.folder + where + which_type + exp_name + '_exp-variance' + '.' + format,
                        bbox_inches="tight"
                    )

                    self.__plot_exp_metrics_sgl(
                        metrics_dict=metrics_dict,
                        comb_key="",
                        which_type=which_type,
                        format=format
                    )
            else:
                raise ValueError('To many parameters for experiment \'' + exp_name + '\', can plot metric variation'
                                                                                 'for 1 or 2 varying params!')
        return exp_plot_data

    def __plot_exp_metrics_sgl(self, metrics_dict, comb_key, which_type, format, where=''):
        metric_keys = [elem.split('-') for elem in metrics_dict.keys()]
        lns = [len(elem) for elem in metric_keys]
        max_iter = max(lns)
        if min(lns) != max_iter:  # first time through
            # divide them based on how many comb of exp params
            for i in range(max_iter, 1, -1):
                keys_to_plot = [key for key in metrics_dict.keys() if len(key.split('-')) == i]
                self.__plot_exp_metrics_sgl(
                    metrics_dict={k: metrics_dict[k] for k in keys_to_plot},
                    comb_key=comb_key,
                    which_type=which_type,
                    format=format
                )
        elif max_iter > 2:  # needs still to iterate
            for i in range(1, max_iter):  #
                popped_keys = np.unique([key.split('-')[i] for key in metrics_dict.keys()])
                for exp_key in popped_keys:
                    self.__plot_exp_metrics_sgl(
                        metrics_dict={'-'.join(k.split('-')[:i]+k.split('-')[i+1:]): metrics_dict[k]
                                      for k in metrics_dict.keys() if k.split('-')[i] == exp_key},
                        comb_key=comb_key + '-' + exp_key,
                        which_type=which_type,
                        format=format
                    )
        else:  # done, plot
            exp_names = np.unique([exp.split('-')[0] for exp in metrics_dict.keys()])
            exp_params = np.unique([exp.split('-')[1][0] for exp in metrics_dict.keys()])
            for exp_name in exp_names:
                for exp_param in exp_params:
                    param = exp_param
                    if exp_param in self.param_descriptions.keys():
                        param = self.param_descriptions[exp_param]
                    xs_str = np.unique([exp.split('-')[1][1:] for exp in metrics_dict.keys()
                                   if exp.split('-')[0] == exp_name
                                   and exp.split('-')[1][0] == exp_param])
                    xs = np.sort([np.int64(xs_str[i]) if all([d.isdigit() for d in xs_str[i]])
                          else np.float64(xs_str[i]) for i in range(len(xs_str))])

                    # SINGLE PARAMETER 'NA'-DISTANCE PLOT
                    na_dist = [
                        metrics_dict[exp_name+'-'+exp_param+str(xs[i])]['na_dist']
                        for i in range(len(xs))
                    ]
                    accuracy_fig = plt.figure(figsize=(13, 8))
                    ax = accuracy_fig.add_subplot(111)
                    ax.set_xlabel(param)
                    ax.set_ylabel('dist to $na$')
                    for i in range(len(xs)):
                        plt.plot(range(len(na_dist[0])), na_dist[i], '-', label=exp_param+str(xs[i]))
                    plt.xticks(
                        range(len(na_dist[0])),
                        [self.target_to_cls_dict[which_type][i] for i in range(len(na_dist[0]))],
                        size='medium'
                     )
                    plt.legend()
                    accuracy_fig.tight_layout()
                    accuracy_fig.savefig(
                        self.folder + where + which_type + '_nadist' + comb_key + '.' + format,
                        bbox_inches="tight"
                    )
                    plt.close(accuracy_fig)

                    # # SINGLE PARAMETER SILHOUETTE PLOT
                    # # following line might give issues if depth/rotation in float.. need to think of conv to string
                    # silhouette_coefficients = [
                    #     metrics_dict[exp_name+'-'+exp_param+str(xs[i])]['silhouette_coefficient']
                    #     for i in range(len(xs))
                    # ]
                    # itr_clsdist_fig = plt.figure(figsize=(13, 8))
                    # ax = itr_clsdist_fig.add_subplot(111)
                    # ax.set_xlabel(param)
                    # ax.set_ylabel('silhouette coefficient')
                    # plt.plot(xs, silhouette_coefficients, '-')
                    # itr_clsdist_fig.tight_layout()
                    # itr_clsdist_fig.savefig(
                    #     self.folder + where + which_type + '_silhouette' + comb_key + '.' + format,
                    #     bbox_inches="tight"
                    # )
                    # plt.close(itr_clsdist_fig)

    def __get_digit(self, string):
        a = str(1)
        return [c for c in string if c.isdigit() or c=='.']

    def get_skin(self):
        return self.skin

    def get_experiment_data_cls_pos(self):
        return self.sectors

    def get_experiment_data(self):
        return self.experiment_data

    def set_experiment_data(self, experiment_data):
        self.experiment_data = experiment_data
        return True

    def get_experiments(self):
        return self.experiments

    def get_sectors(self):
        return self.sectors

    def get_class_names(self):
        return self.class_names

    def set_class_names(self, class_names):
        self.class_names = class_names
        for key in self.targets.keys():
            self.targets[key] = self._get_targets(key)
        return True

    def get_targets(self):
        return self.targets

    def get_class_type(self, key=None):
        if key is not None:
            return self.class_type[key]
        return self.class_type['properties']

    def set_class_type(self, class_type, key=None):
        if key is not None:
            self.class_type[key] = class_type
        else:
            self.class_type = class_type
        for key in self.class_type.keys():
            self.targets[key] = self._get_targets(which=key)
        return True


def plot_3d_params(xs, ys, zs, label_descriptions,
                   which_type='properties', color_map=c_map.Blues):
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.view_init(azim=30, elev=55)
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    if which_type == 'presence':
        ax.view_init(azim=-80)
    ax.set_xlabel(label_descriptions[0], fontsize=16)
    ax.set_ylabel(label_descriptions[1], fontsize=16)
    ax.set_zlabel(label_descriptions[2], fontsize=16)
    xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
    surf = ax.plot_trisurf(xs, ys, zs,
                           cmap=color_map,
                           linewidth=.6,
                           antialiased=False,
                           alpha=1,
                           edgecolors='k')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig


def histogram_plot(hist_array, xlabel='$principal components$', ylabel='$explained variance (%)$', xtick_prefix='p'):
    # init figure
    hist_fig = plt.figure(figsize=(13, 8))
    ax = hist_fig.add_subplot(111)

    # necessary variables
    ind = np.arange(1, len(hist_array) + 1)  # the x locations for the groups
    width = 0.85

    # the bars
    ax.bar(ind, hist_array * 100, width, color='black')

    xTickMarks = ['$' + xtick_prefix + '_{' + str(i) + '}$'  for i in ind]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=34)

    return hist_fig


def plotbdbs(sectors=['2-2', '3-2'], data=None):
    # NOTE: hard coded!!

    X = data['X']
    markers = data['markers']
    targets = data['targets']
    cls_to_target_dict = data['cls_to_target_dict']
    tactile_sectors = data['tactile_sectors']
    class_names = data['class_names']
    target_to_cls_dict = data['target_to_cls_dict']
    ax_cpy = data['ax']

    # Create color maps
    colors_bold = ['#0191c4', '#b71500', '#d1d800', '#00b235', '#3400af']
    cmap_bold = ListedColormap(colors_bold)

    # plot mash
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax.set_ylabel('$\\vec{p}_2$', fontsize=48)

    # plot data points
    for sect in sectors:
        ind = [i for i in range(len(tactile_sectors)) if tactile_sectors[i] == sect][0]
        ax.scatter(
            X[ind, 0], X[ind, 1],
            c=targets[ind],
            cmap=cmap_bold,
            marker=markers[targets[ind]],
            edgecolor='k',
            s=640,
            vmin=0,
            vmax=len(colors_bold),
            alpha=0.7
        )

    xlim = list(ax_cpy.get_xlim())
    ylim = list(ax_cpy.get_ylim())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # add legends
    patches = []
    for i in [cls_to_target_dict[class_names[sect]] for sect in sectors]:
        patches.append(
            plt.plot(
                [], [],
                marker=markers[i],
                ms=20,
                ls="",
                markeredgecolor='k',
                mec=None,
                color=colors_bold[i],
                label=target_to_cls_dict[i])[0]
        )
    ax.legend(handles=patches, fontsize=22, loc='best')
    return fig