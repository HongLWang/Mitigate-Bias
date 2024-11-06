from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.linear_model import LogisticRegression
from utils import split_data_trn_vld_tst
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing import DisparateImpactRemover


def build_logit_model_n_mesure_bias(dset_trn,dset_tst, privileged_groups, unprivileged_groups):

    scaler = StandardScaler()
    X_trn = scaler.fit_transform(dset_trn.features)
    y_trn = dset_trn.labels.ravel()
    w_trn = dset_trn.instance_weights.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_trn, y_trn, sample_weight = w_trn)

    dset_tst_pred = dset_tst.copy(deepcopy=True)
    X_tst = scaler.transform(dset_tst_pred.features)
    y_tst = dset_tst_pred.labels
    dset_tst_pred.scores = lmod.predict_proba(X_tst)[:, 0].reshape(-1,1)

    fav_inds = np.where(lmod.predict(X_tst) == dset_trn.favorable_label)[0]

    dset_tst_pred.labels[fav_inds] = dset_tst_pred.favorable_label

    dset_tst_pred.labels[~fav_inds] = dset_tst_pred.unfavorable_label

    metric_tst = ClassificationMetric(dset_tst, dset_tst_pred, unprivileged_groups, privileged_groups)

    predictive_parity = metric_tst.positive_predictive_value(privileged=True)-metric_tst.positive_predictive_value(privileged=False)
    equalized_odds = metric_tst.equalized_odds_difference()

    print('predictive_parity')
    print(predictive_parity)
    print('equalized_odds')
    print(equalized_odds)

    overall_acc = metric_tst.performance_measures(privileged=None)['ACC']

    privileged_acc = metric_tst.performance_measures(privileged=True)['ACC']
    unprivileged_acc = metric_tst.performance_measures(privileged=False)['ACC']

    print('overall_acc, privileged_acc, unprivileged_acc')
    print(overall_acc, privileged_acc, unprivileged_acc)


def check_bias(attribute_name,preproc_compas):
    # Define privileged and unprivileged groups
    priv_group = [{attribute_name: 1}]  # Male
    unpriv_group = [{attribute_name: 0}]  # Female

    # Load preprocessed COMPAS data

    # Calculate the statistical parity difference
    compas_metrics = BinaryLabelDatasetMetric(preproc_compas, unprivileged_groups=unpriv_group, privileged_groups=priv_group)
    print("Statistical Parity Difference:", compas_metrics.statistical_parity_difference())


    # Obtain the split datasets
    dset_raw_trn, dset_raw_vld, dset_raw_tst = split_data_trn_vld_tst(preproc_compas, priv_group, unpriv_group)

    build_logit_model_n_mesure_bias(dset_raw_trn,dset_raw_tst, priv_group, unpriv_group)



if __name__ == '__main__':
    attribute_names = ['race','sex']

    for attributes in attribute_names:
        print('\n\n working on attribute {}'.format(attributes))
        preproc_compas = load_preproc_data_compas([attributes])

        check_bias(attributes,preproc_compas)