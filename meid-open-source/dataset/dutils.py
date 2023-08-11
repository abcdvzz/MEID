
# FEATURE_NAME: PATH_TO_FEATURE

path_dict = {
        "ResNet101": '/data/lxj/data/videolt/r101/ResNet101-feature',
        "TSM-R50": '/data/lxj/data/videolt/tsm-r50/TSM-R50-feature',
        "ResNet50": '/data/lxj/data/videolt/r50/ResNet50-feature',
        "Charades": '/data/lxj/data/charades/charades_lt_r101',
        "CharadesEgo": '/data/lxj/data/CharadesEgo/r101'
        }

# FEATURE_DIM
dim_dict = {
            "ResNet101": 2048,
            "TSM-R50": 2048,
            "ResNet50": 2048,
            "Charades": 2048,
            "CharadesEgo": 2048
            }

def get_feature_path(feature_name):
    return path_dict[feature_name]

def get_feature_dim(feature_name):
    return dim_dict[feature_name]

# PATH_TO_LABELS
ROOT='./labels'

def get_label_path(dataset='VideoLT'):
    
    if dataset == 'Charades':
        root = '/data/lxj/data/charades/'
        lc_list = root+'count-labels-train.lst'
        train_list = root+'train_lt.txt'
        val_list = root+'test_lt.txt'
        return lc_list, train_list, val_list
    if dataset == 'CharadesEgo':
        root = '/data/lxj/data/CharadesEgo/'
        lc_list = root+'count-labels-train.lst'
        train_list = root+'train_lt.txt'
        val_list = root+'test_lt.txt'
        return lc_list, train_list, val_list
    
    lc_list = ROOT+'/count-labels-train.lst'
    train_list = ROOT+'/train.lst'
    val_list = ROOT+'/test.lst'
    return lc_list, train_list, val_list


def get_lt_plus_path(dataset='VideoLT'):

    if dataset == 'Charades':
        root = '/data/lxj/data/charades/'
        lc_list = root+'count-labels-train.lst'
        train_list = root+'train_lt.txt'
        # val_list = root+'test_lt.txt'
        val_list = root+'val_lt.txt'
        return lc_list, train_list, val_list
    if dataset == 'CharadesEgo':
        root = '/data/lxj/data/CharadesEgo/'
        lc_list = root+'count-labels-train.lst'
        train_list = root+'train_lt.txt'
        # val_list = root+'test_lt.txt'
        val_list = root+'val_lt.txt'
        return lc_list, train_list, val_list

    lc_list = ROOT+'/count-labels-train.lst'
    # train_list = ROOT+'/train_plus_pred.lst'
    # val_list = ROOT+'/test_plus_pred.lst'
    train_list = ROOT+'/train_plus_lt.lst'
    val_list = ROOT+'/test_plus_lt.lst'
    return lc_list, train_list, val_list
