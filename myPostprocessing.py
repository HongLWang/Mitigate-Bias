# %matplotlib inline
# Load all necessary packages
import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


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

    print('priviliaged group')
    print(f'PPR_p = {PPR_p}, FNR_p = {FNR_p}, FPR_p = {FPR_p}, PPV_p = {PPV_p}, FPV_p = {FPV_p} \n')

    print('unpriviliaged group')
    print(f'PPR_up = {PPR_up}, FNR_up = {FNR_up}, FPR_up = {FPR_up}, PPV_up = {PPV_up}, FPV_up = {FPV_up} \n')


dataset_orig = load_preproc_data_compas()

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

def try_calibrated_odds():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve

    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Placeholder for predicted and transformed datasets
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

    # Logistic regression classifier and predictions for training data
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)

    fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

    # Prediction probs for validation and testing data
    X_valid = scale_orig.transform(dataset_orig_valid.features)
    y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

    X_test = scale_orig.transform(dataset_orig_test.features)
    y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
    dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred

    y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
    dataset_orig_valid_pred.labels = y_valid_pred

    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred

    cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

    matrixP = cm_pred_test.binary_confusion_matrix(privileged=True)

    matrixUP = cm_pred_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, cm_pred_test)



    # apply equalized odd

    cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups,
                                         cost_constraint='fnr',
                                         seed=123456)
    cpp = cpp.fit(dataset_orig_train, dataset_orig_train_pred)


    dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

    cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)

    matrixP = cm_transf_test.binary_confusion_matrix(privileged=True)

    matrixUP = cm_transf_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, cm_transf_test)



def try_equalized_odds():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve

    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Placeholder for predicted and transformed datasets
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

    # Logistic regression classifier and predictions for training data
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)

    fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

    # Prediction probs for validation and testing data
    X_valid = scale_orig.transform(dataset_orig_valid.features)
    y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

    X_test = scale_orig.transform(dataset_orig_test.features)
    y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
    dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred

    y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
    dataset_orig_valid_pred.labels = y_valid_pred

    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred

    cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

    matrixP = cm_pred_test.binary_confusion_matrix(privileged=True)

    matrixUP = cm_pred_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, cm_pred_test)



    # apply equalized odd

    cpp = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups,
                                         seed=123456)
    cpp = cpp.fit(dataset_orig_train, dataset_orig_train_pred)


    dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

    cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)

    matrixP = cm_transf_test.binary_confusion_matrix(privileged=True)

    matrixUP = cm_transf_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, cm_transf_test)


# try_calibrated_odds()
try_equalized_odds()