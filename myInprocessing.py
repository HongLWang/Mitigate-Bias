# %matplotlib inline
# Load all necessary packages
import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt

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

def try_adversarial_debiasing():

    # train a plain classifier with no bias-mitigation.
    sess = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='plain_classifier',
                              debias=False,
                              sess=sess)


    plain_model.fit(dataset_orig_train )


    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

    # Metrics for the dataset from plain model (without debiasing)
    print("#### Plain model - without debiasing - dataset metrics")
    metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

    metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    print("Independence: Difference in positive prediction between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

    print("#### Plain model - without debiasing - classification metrics")
    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                     dataset_nodebiasing_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    matrixP = classified_metric_nodebiasing_test.binary_confusion_matrix(privileged=True)

    matrixUP = classified_metric_nodebiasing_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, classified_metric_nodebiasing_test)


    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()


    # Learn parameters with debias set to True, apply adversial debiasing
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)


    debiased_model.fit(dataset_orig_train)
    # Apply the plain model to test data
    dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                     dataset_debiasing_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)



    matrixP = classified_metric_debiasing_test.binary_confusion_matrix(privileged=True)

    matrixUP = classified_metric_debiasing_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, classified_metric_debiasing_test)


def try_prejudice_remover():
    # train a plain classifier with no bias-mitigation.
    plain_model = PrejudiceRemover(eta=0, sensitive_attr="race").fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

    # Metrics for the dataset from plain model (without debiasing)


    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                              dataset_nodebiasing_test,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)

    matrixP = classified_metric_nodebiasing_test.binary_confusion_matrix(privileged=True)

    matrixUP = classified_metric_nodebiasing_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, classified_metric_nodebiasing_test)



    # Learn parameters with debias set to True, apply adversial debiasing
    debiased_model = PrejudiceRemover(eta=1, sensitive_attr="race").fit(dataset_orig_train)

    # debiased_model.fit(dataset_orig_train)
    # Apply the plain model to test data
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                            dataset_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

    matrixP = classified_metric_debiasing_test.binary_confusion_matrix(privileged=True)

    matrixUP = classified_metric_debiasing_test.binary_confusion_matrix(privileged=False)

    calculate_three_metric(matrixP, matrixUP, classified_metric_debiasing_test)


# try_adversarial_debiasing()
try_prejudice_remover()