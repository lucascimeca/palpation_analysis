from experiment_setup import ExperimentSetup

SHOW = False
SAVE_LOCAL = False
SAVE_GLOBAL = True
RESOLUTION = 1000
SAVE_FORMAT = 'png'
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
    data_path = {
        'phantom1': "./../data/presence_exp.mat",
        'phantom2': "./../data/properties_exp.mat"
    }

    exp_metric_data = dict()
    exp_fig_data = dict()
    exp_setup = dict()

    for env in ['phantom2']: #properties_exp_state.keys():
        exp_setup[env] = ExperimentSetup(data_path[env], baseline="./../data/baseline.mat")
        exp_setup[env].set_class_names(properties_exp_state[env])
        exp_metric_data[env], exp_fig_data[env] = exp_setup[env].run_experiments(
            which_type='properties',
            which_clustering='k_means',
            show=SHOW,
            save_local=SAVE_LOCAL,
            save_global=SAVE_GLOBAL,
            where=env,
            resolution=RESOLUTION,
            format=SAVE_FORMAT
            )
