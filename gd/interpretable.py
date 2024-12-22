import dill as pickle
from data_loaders import get_datasets, CATEGORICAL_FEATURES
from policy import RURAL_COVARIATE_LIST, COVARIATE_LIST
import numpy as np
import itertools
from reporting import write_result


def get_num_cats(cat_feat, features):
    bs = []
    for i, w in enumerate(features):
        if w.startswith(cat_feat):
            bs.append(i)
    return len(bs)


def map_one_hot_to_name(features, cats, one_hot):
    name_dic = {}

    idx = 0
    for i, feature in enumerate(features):
        if one_hot[i] == 1:
            name_dic[cats[idx]] = feature.removeprefix("{}_".format(cats[idx]))
            idx += 1
    return name_dic


def generate_one_hot_vectors(categories, subcategories, num_categories):
    indices = [np.eye(num) for num in num_categories]
    index_combinations = itertools.product(*indices)
    one_hot_vectors = []
    names = []
    for combo in index_combinations:
        vector = []
        for i in range(len(categories)):
            vector.extend(list(combo[i]))
        one_hot_vectors.append(vector)
        names.append(map_one_hot_to_name(subcategories, RURAL_COVARIATE_LIST, vector))
    return np.array(one_hot_vectors), names


def update_namedic_with_transfer_values(name_dics, transfers):
    for i, name_dic in enumerate(name_dics):
        tx = 0
        for tup in transfers[i]:
            tx += tup[0] * tup[1]

        name_dic["expected_transfer"] = tx
    return name_dics


def export_namedics_to_csv(filename, name_dics):
    for name_dic in name_dics:
        write_result("{}".format(filename), name_dic)


def export_policy(district, policy_type, uncondtol):
    def load_policy(file_path):
        with open(file_path, "rb") as f:
            t = pickle.load(f)
        return t

    t = load_policy(
        "policies/{}_{}_uncondtol={}.pickle".format(district, policy_type, uncondtol)
    )

    if district == "all":
        covariates = COVARIATE_LIST
    else:
        covariates = RURAL_COVARIATE_LIST

    _, _, features = get_datasets(district, pool="rural", covariates=covariates)

    num_cats = [get_num_cats(cat_feat, features) for cat_feat in covariates]

    one_hot_vectors, names = generate_one_hot_vectors(covariates, features, num_cats)
    name_dics = update_namedic_with_transfer_values(names, t(one_hot_vectors))

    export_namedics_to_csv(
        "district_policy_tables/{}_{}_uncondtol={}.csv".format(
            district, policy_type, uncondtol
        ),
        name_dics,
    )


for district in ["mchinji", "dowa", "kasungu", "karonga", "chitipa"]:
    for policy in ["saturation", "binary"]:
        for uncondtol in [0.1, 0.2]:
            export_policy(district, policy, uncondtol)
