from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import LFR
from aif360.metrics import BinaryLabelDatasetMetric
from utils import split_data_trn_vld_tst
from Measure_Bias import build_logit_model_n_mesure_bias
from aif360.algorithms.preprocessing import DisparateImpactRemover
import numpy as np
from sklearn.linear_model import LogisticRegression


def reweight(attribute_name):
    preproc_compas = load_preproc_data_compas([attribute_name])
    priv_group = [{attribute_name: 1}]
    unpriv_group = [{attribute_name: 0}]

    # Obtain the split datasets
    dset_raw_trn, dset_raw_vld, dset_raw_tst = split_data_trn_vld_tst(preproc_compas, priv_group, unpriv_group)
    RW = Reweighing(privileged_groups=priv_group, unprivileged_groups=unpriv_group)
    RW.fit(dset_raw_trn)
    dset_rewgt_trn = RW.transform(dset_raw_trn)
    metric_rewgt_trn = BinaryLabelDatasetMetric(dset_rewgt_trn, unprivileged_groups=unpriv_group,
                                                privileged_groups=priv_group)

    print("Difference in mean outcomes = %f" % metric_rewgt_trn.mean_difference())
    print("Disparate impact = %f" % metric_rewgt_trn.disparate_impact())

    build_logit_model_n_mesure_bias(dset_rewgt_trn,dset_raw_tst, priv_group, unpriv_group)



def remove_disparate_impact(attribute_name):
    preproc_compas = load_preproc_data_compas([attribute_name])
    priv_group = [{attribute_name: 1}]
    unpriv_group = [{attribute_name: 0}]

    # Obtain the split datasets
    dset_raw_trn, dset_raw_vld, dset_raw_tst = split_data_trn_vld_tst(preproc_compas, priv_group, unpriv_group)
    index = dset_raw_trn.feature_names.index(attribute_name)

    DIs = []
    for level in [0.5]:
        di = DisparateImpactRemover(repair_level=level)
        train_repd = di.fit_transform(dset_raw_trn)
        test_repd = di.fit_transform(dset_raw_tst)

        train_repd.features = np.delete(train_repd.features, index, axis=1)
        test_repd.features = np.delete(test_repd.features, index, axis=1)

        build_logit_model_n_mesure_bias(train_repd, test_repd, priv_group, unpriv_group)


def learn_fair_representation(attribute_name):
    preproc_compas = load_preproc_data_compas([attribute_name])
    priv_group = [{attribute_name: 1}]
    unpriv_group = [{attribute_name: 0}]

    # Obtain the split datasets
    dset_raw_trn, dset_raw_vld, dset_raw_tst = split_data_trn_vld_tst(preproc_compas, priv_group, unpriv_group)

    TR = LFR(unprivileged_groups=unpriv_group, privileged_groups=priv_group)
    TR = TR.fit(dset_raw_trn)

    dset_lfr_trn = TR.transform(dset_raw_trn, threshold=thresholds[attribute_name]) # 0.55 for race, and
    dset_lfr_trn = dset_raw_trn.align_datasets(dset_lfr_trn)

    # dset_lfr_tst = TR.transform(dset_raw_tst, threshold=0.7)
    # dset_lfr_tst = dset_raw_trn.align_datasets(dset_lfr_tst)

    build_logit_model_n_mesure_bias(dset_lfr_trn, dset_raw_tst, priv_group, unpriv_group)

if __name__ == '__main__':
    attribute_name_list = ['race','sex']
    thresholds = {'race':0.55, 'sex':0.45}
    for attribute_name in attribute_name_list:
        print('\n\n working on attribute {}'.format(attribute_name))

        reweight(attribute_name)
        remove_disparate_impact(attribute_name)
        learn_fair_representation(attribute_name)

