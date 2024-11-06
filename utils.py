
# Split data into training, validation, and test sets
def split_data_trn_vld_tst(data, priv_group, unpriv_group, train_ratio=0.6, val_ratio=0.2):
    data_train, data_val, data_test = data.split([train_ratio, val_ratio], shuffle=True, seed = 42)
    return data_train, data_val, data_test

