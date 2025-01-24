from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.sklearn.metrics import statistical_parity_difference
from sklearn.linear_model import LogisticRegression
from utils import split_data_trn_vld_tst
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from aif360.algorithms.preprocessing.reweighing import Reweighing

from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing



def calculate_three_metric(matrixP, matrixUP, metric):

    cnt_p = matrixP['TP'] + matrixP['FP'] + matrixP['TN'] + matrixP['FN']
    PPR_p = (matrixP['TP'] + matrixP[
        'FP']) / cnt_p  # independence metric, positive prediction rate of priviliaged group


    cnt_up = matrixUP['TP'] + matrixUP['FP'] + matrixUP['TN'] + matrixUP['FN']
    PPR_up = (matrixUP['TP'] + matrixUP[
        'FP']) / cnt_up  # independence metric positive prediction rate of unpriviliaged group


    FNR_p = metric.false_negative_rate(privileged=True)
    FPR_p = metric.false_positive_rate(privileged=True)  # privileged FNR and FPR are for seperation

    FNR_up = metric.false_negative_rate(privileged=False)
    FPR_up = metric.false_positive_rate(privileged=False)  # unprivileged FNR and FPR are for seperation

    PPV_p = metric.positive_predictive_value(privileged=True)  # privileged, positive predictive value, for sufficiency
    FPV_p = metric.negative_predictive_value(privileged=True)  # privileged,  negative predictive value, for sufficiency

    PPV_up = metric.positive_predictive_value(
        privileged=False)  # privileged, positive predictive value, for sufficiency
    FPV_up = metric.negative_predictive_value(
        privileged=False)  # privileged,  negative predictive value, for sufficiency

    # print('priviliaged group')
    # print(f'PPR_p = {PPR_p}, FNR_p = {FNR_p}, FPR_p = {FPR_p}, PPV_p = {PPV_p}, FPV_p = {FPV_p} \n')
    #
    # print('unpriviliaged group')
    # print(f'PPR_up = {PPR_up}, FNR_up = {FNR_up}, FPR_up = {FPR_up}, PPV_up = {PPV_up}, FPV_up = {FPV_up} \n')

    return FPR_p, FPR_up

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


def Statistical_Parity_Subgroup_Fairness(preproc_compas):

    # Define privileged and unprivileged groups
    priv_group = [{'sex':1,'race':1}]
    unpriv_group = [{'sex': 0,'race':0}]  # Female

    compas_metrics = BinaryLabelDatasetMetric(preproc_compas, privileged_groups=priv_group, unprivileged_groups=unpriv_group)
    statistical_parity = compas_metrics.statistical_parity_difference()
    # print('statistical_parity')
    # print(statistical_parity)
    return statistical_parity


def False_Positive_Rate_Subgroup_Fairness(preproc_compas):

    # Define privileged and unprivileged groups
    priv_group = [{'sex':1,'race':1}]
    unpriv_group = [{'sex': 0,'race':0}]  # Female


    dataset_orig_train, dataset_orig_test = preproc_compas.split([0.7], shuffle=True)
    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    # train a plain classifier with no bias-mitigation.
    plain_model = PrejudiceRemover(eta=0, sensitive_attr="race").fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

    # Metrics for the dataset from plain model (without debiasing)

    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                              dataset_nodebiasing_test,
                                                              unprivileged_groups=unpriv_group,
                                                              privileged_groups=priv_group)

    matrixP = classified_metric_nodebiasing_test.binary_confusion_matrix(privileged=True)

    matrixUP = classified_metric_nodebiasing_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, classified_metric_nodebiasing_test)

    FPR_p, FPR_up = calculate_three_metric(matrixP, matrixUP, classified_metric_nodebiasing_test)

    return FPR_p, FPR_up

def Multicalibration_Fairness(preproc_compas):

    # label=0, attribute = privillaged sex + privillaged race, Probability of getting prediction =0?
    priv_group = [{'sex': 1, 'race': 1}]



    dataset_orig_train, dataset_orig_test = preproc_compas.split([0.7], shuffle=True)
    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    # train a plain classifier with no bias-mitigation.
    plain_model = PrejudiceRemover(eta=0).fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)



    num_privSex_privRace = sum( [1 for feature in dataset_orig_test.protected_attributes if np.all(feature == [1,1])])

    for v in [0,1]:
        # any certain prediction, sex=1 and race = 1
        num_Pos_label_privSex_privRace = sum(1 for score, feature in zip(dataset_nodebiasing_test.labels, dataset_nodebiasing_test.protected_attributes)
                                             if score == v and np.all(feature == [1,1]))

        # any certain ground truth, sex=1 and race = 1
        num_Pos_GT_privSex_privRace = sum(
            1 for label, feature in zip(dataset_orig_test.labels, dataset_orig_test.protected_attributes)
            if label == v and np.all(feature == [1,1]))

        difference =  (num_Pos_GT_privSex_privRace - num_Pos_label_privSex_privRace)/num_privSex_privRace

        print('multicalibration fairness for ', ['favorable output is ' if v == 0 else 'unfavorable output is '], difference)




    unpriv_group = [{'sex': 0, 'race': 0}]
    RW = Reweighing(privileged_groups=priv_group, unprivileged_groups=unpriv_group)
    RW.fit(dataset_orig_train)
    dset_rewgt_trn = RW.transform(dataset_orig_test)


    # train a plain classifier with no bias-mitigation.
    plain_model = PrejudiceRemover(eta=0).fit(dset_rewgt_trn)

    # Apply the plain model to test data
    PrejudiceRemover_test = plain_model.predict(dset_rewgt_trn)

    fairness = Statistical_Parity_Subgroup_Fairness(dset_rewgt_trn)
    print('statistical parity fairness after reweighting', fairness)

    fpr_p, fpr_up = False_Positive_Rate_Subgroup_Fairness(dset_rewgt_trn)
    print('FPR_p, fpr_up after reweighting', fpr_p, fpr_up)
    num_privSex_privRace = sum([1 for feature in dataset_orig_test.protected_attributes if np.all(feature == [1, 1])])

    for v in [0, 1]:
        # any certain prediction, sex=1 and race = 1
        num_Pos_label_privSex_privRace = sum(
            1 for score, feature in zip(PrejudiceRemover_test.labels, PrejudiceRemover_test.protected_attributes)
            if score == v and np.all(feature == [1, 1]))

        # any certain ground truth, sex=1 and race = 1
        num_Pos_GT_privSex_privRace = sum(
            1 for label, feature in zip(dataset_orig_test.labels, dataset_orig_test.protected_attributes)
            if label == v and np.all(feature == [1, 1]))

        difference = (num_Pos_GT_privSex_privRace - num_Pos_label_privSex_privRace) / num_privSex_privRace

        print('multicalibration fairness after reweighting for ', ['favorable output is ' if v == 0 else 'unfavorable output is '],
              difference)

    # apply equalized odd

    cpp = EqOddsPostprocessing(privileged_groups=priv_group,
                                         unprivileged_groups=unpriv_group,
                                         seed=123456)
    cpp = cpp.fit(dataset_orig_test, dataset_nodebiasing_test)


    dataset_transf_test_pred = cpp.predict(dataset_nodebiasing_test)

    statistical_parity = Statistical_Parity_Subgroup_Fairness(dataset_transf_test_pred)
    print('statistical_parity after equalied odd is ', statistical_parity )
    fpr_p, fpr_up = False_Positive_Rate_Subgroup_Fairness(dataset_transf_test_pred)
    print('FPR_p, fpr_up after equalized odd', fpr_p, fpr_up)


    num_privSex_privRace = sum([1 for feature in dataset_orig_test.protected_attributes if np.all(feature == [1, 1])])

    for v in [0, 1]:
        # any certain prediction, sex=1 and race = 1
        num_Pos_label_privSex_privRace = sum(
            1 for score, feature in zip(dataset_transf_test_pred.labels, dataset_transf_test_pred.protected_attributes)
            if score == v and np.all(feature == [1, 1]))

        # any certain ground truth, sex=1 and race = 1
        num_Pos_GT_privSex_privRace = sum(
            1 for label, feature in zip(dataset_orig_test.labels, dataset_orig_test.protected_attributes)
            if label == v and np.all(feature == [1, 1]))

        difference = (num_Pos_GT_privSex_privRace - num_Pos_label_privSex_privRace) / num_privSex_privRace

        print('multicalibration fairness after equalized odds for ', ['favorable output is ' if v == 0 else 'unfavorable output is '],
              difference)





def Differential_Fairness(preproc_compas):

    # label=0, attribute = privillaged sex + privillaged race, Probability of getting prediction =0?
    priv_group = [{'sex': 1, 'race': 1}]
    unpriv_group = [{'sex': 0, 'race': 0}]



    dataset_orig_train, dataset_orig_test = preproc_compas.split([0.7], shuffle=True)
    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    # train a plain classifier with no bias-mitigation.
    plain_model = PrejudiceRemover(eta=0).fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)


    num_privSex_privRace = sum( [1 for feature in dataset_orig_test.protected_attributes if np.all(feature == [1,1])])
    num_unprivSex_privRace = sum( [1 for feature in dataset_orig_test.protected_attributes if not np.all(feature == [1,1])])


    for v in [0,1]:
        # any certain prediction, sex=1 and race = 1
        num_Pos_label_privSex_privRace = sum(1 for score, feature in zip(dataset_nodebiasing_test.labels, dataset_nodebiasing_test.protected_attributes)
                                             if score == v and np.all(feature == [1,1]))

        # any certain ground truth, sex=1 and race = 1
        num_Pos_label_unprivSex_privRace = sum(
            1 for label, feature in zip(dataset_nodebiasing_test.labels, dataset_nodebiasing_test.protected_attributes)
            if label == v and not np.all(feature == [1,1]))

        differential =  num_Pos_label_unprivSex_privRace/num_Pos_label_privSex_privRace/num_unprivSex_privRace *num_privSex_privRace

        print('differential fairness for ', ['favorable output is ' if v == 0 else 'unfavorable output is '], differential)






    unpriv_group = [{'sex': 0, 'race': 0}]
    RW = Reweighing(privileged_groups=priv_group, unprivileged_groups=unpriv_group)
    RW.fit(dataset_orig_test)
    dset_rewgt_trn = RW.transform(dataset_orig_test)

    # train a plain classifier with no bias-mitigation.
    plain_model = PrejudiceRemover(eta=0).fit(dset_rewgt_trn)

    # Apply the plain model to test data
    PrejudiceRemover_test = plain_model.predict(dset_rewgt_trn)

    num_privSex_privRace = sum([1 for feature in dataset_orig_test.protected_attributes if np.all(feature == [1, 1])])

    for v in [0, 1]:
        # any certain prediction, sex=1 and race = 1
        num_Pos_label_privSex_privRace = sum(
            1 for score, feature in zip(PrejudiceRemover_test.labels, PrejudiceRemover_test.protected_attributes)
            if score == v and np.all(feature == [1, 1]))

        # any certain ground truth, sex=1 and race = 1
        num_Pos_label_unprivSex_privRace = sum(
            1 for label, feature in zip(PrejudiceRemover_test.labels, PrejudiceRemover_test.protected_attributes)
            if label == v and not np.all(feature == [1, 1]))

        differential = num_Pos_label_unprivSex_privRace / num_Pos_label_privSex_privRace / num_unprivSex_privRace * num_privSex_privRace



        print('differencial fairness after reweighting for ', ['favorable output is ' if v == 0 else 'unfavorable output is '],
              differential)

    # apply equalized odd

    cpp = EqOddsPostprocessing(privileged_groups=priv_group,
                                         unprivileged_groups=unpriv_group,
                                         seed=123456)
    cpp = cpp.fit(dataset_orig_test, dataset_nodebiasing_test)


    dataset_transf_test_pred = cpp.predict(dataset_nodebiasing_test)


    num_privSex_privRace = sum([1 for feature in dataset_orig_test.protected_attributes if np.all(feature == [1, 1])])

    for v in [0, 1]:
        # any certain prediction, sex=1 and race = 1
        num_Pos_label_privSex_privRace = sum(
            1 for score, feature in zip(dataset_transf_test_pred.labels, dataset_transf_test_pred.protected_attributes)
            if score == v and np.all(feature == [1, 1]))

        # any certain ground truth, sex=1 and race = 1
        num_Pos_label_unprivSex_privRace = sum(
            1 for label, feature in zip(dataset_transf_test_pred.labels, dataset_transf_test_pred.protected_attributes)
            if label == v and not np.all(feature == [1, 1]))

        differential = num_Pos_label_unprivSex_privRace / num_Pos_label_privSex_privRace / num_unprivSex_privRace * num_privSex_privRace

        print('differencial fairness after equalized odds for ', ['favorable output is ' if v == 0 else 'unfavorable output is '],
              differential)








if __name__ == '__main__':

    preproc_compas = load_preproc_data_compas()

    Multicalibration_Fairness(preproc_compas)
    Differential_Fairness(preproc_compas)

