import inspect
from src.clusters import (
    FcmRuleCluster,
    FcmMFsCluster,
    KMeansRuleCluster,
    KMeansMFsCluster,
    MBKRuleCluster,
    MBKMFsCluster,
)
from src.fis.fuzzy_logic.mfs.type_2.t2_gaussian_mf import T2GaussianMFBuilder

param_display = {
    "m": "m (fuzziness)",
    "error": "error (tolerance)",
    "maxiter": "maxiter (max iterations)",
    "random_state": "random_state",
    "init": "init",
    "n_init": "n_init (number of runs)",
    "max_iter": "max_iter (max iterations)",
    "tol": "tol (tolerance)",
    "batch_size": "batch_size",
    "max_no_improvement": "max_no_improvement",
    "min_std_ratio": "min_std_ratio",
    "uncertainty_factor": "uncertainty_factor",
    "decimal_places": "decimal_places",
}


def unify_defaults(rule_dict, mfs_dict):
    if rule_dict.keys() != mfs_dict.keys():
        raise ValueError("Different parameter keys detected!")

    unified = {}
    for k in rule_dict:
        v1, v2 = rule_dict[k], mfs_dict[k]
        if v1 != v2:
            raise ValueError(f"Default mismatch for param '{k}': {v1} != {v2}")
        unified[k] = v1

    return unified


def expand_rows(alg_name, params):
    rows_algorithm = [alg_name] + [""] * (len(params) - 1)
    rows_params = [param_display[p] for p in params]
    rows_values = [params[p].default for p in params]  # parameters is inspect.Parameter
    return rows_algorithm, rows_params, rows_values


def get_default_params():
    fcm_rule_defaults = inspect.signature(FcmRuleCluster.__init__).parameters
    fcm_mfs_defaults = inspect.signature(FcmMFsCluster.__init__).parameters
    kmeans_rule_defaults = inspect.signature(KMeansRuleCluster.__init__).parameters
    kmeans_mfs_defaults = inspect.signature(KMeansMFsCluster.__init__).parameters
    mbk_rule_defaults = inspect.signature(MBKRuleCluster.__init__).parameters
    mbk_mfs_defaults = inspect.signature(MBKMFsCluster.__init__).parameters
    t2_gaussian_mf_defaults = inspect.signature(T2GaussianMFBuilder.__init__).parameters

    fcm_rule_defaults = {
        "m": fcm_rule_defaults["m"],
        "error": fcm_rule_defaults["error"],
        "maxiter": fcm_rule_defaults["maxiter"],
        "random_state": fcm_rule_defaults["random_state"],
    }

    fcm_mfs_defaults = {
        "m": fcm_mfs_defaults["m"],
        "error": fcm_mfs_defaults["error"],
        "maxiter": fcm_mfs_defaults["maxiter"],
        "random_state": fcm_mfs_defaults["random_state"],
    }

    kmeans_rule_defaults = {
        "init": kmeans_rule_defaults["init"],
        "n_init": kmeans_rule_defaults["n_init"],
        "max_iter": kmeans_rule_defaults["max_iter"],
        "tol": kmeans_rule_defaults["tol"],
        "random_state": kmeans_rule_defaults["random_state"],
    }

    kmeans_mfs_defaults = {
        "init": kmeans_rule_defaults["init"],
        "n_init": kmeans_rule_defaults["n_init"],
        "max_iter": kmeans_rule_defaults["max_iter"],
        "tol": kmeans_rule_defaults["tol"],
        "random_state": kmeans_rule_defaults["random_state"],
    }

    mbk_rule_defaults = {
        "init": mbk_rule_defaults["init"],
        "n_init": mbk_rule_defaults["n_init"],
        "max_iter": mbk_rule_defaults["max_iter"],
        "tol": mbk_rule_defaults["tol"],
        "batch_size": mbk_rule_defaults["batch_size"],
        "max_no_improvement": mbk_rule_defaults["max_no_improvement"],
        "random_state": mbk_rule_defaults["random_state"],
    }

    mbk_mfs_defaults = {
        "init": mbk_rule_defaults["init"],
        "n_init": mbk_rule_defaults["n_init"],
        "max_iter": mbk_rule_defaults["max_iter"],
        "tol": mbk_rule_defaults["tol"],
        "batch_size": mbk_rule_defaults["batch_size"],
        "max_no_improvement": mbk_rule_defaults["max_no_improvement"],
        "random_state": mbk_mfs_defaults["random_state"],
    }

    alg_pairs = {
        "fcm": (fcm_rule_defaults, fcm_mfs_defaults),
        "kmeans": (kmeans_rule_defaults, kmeans_mfs_defaults),
        "mbk": (mbk_rule_defaults, mbk_mfs_defaults),
    }

    alg_defaults = {
        alg: unify_defaults(rule, mfs) for alg, (rule, mfs) in alg_pairs.items()
    }

    type_2_gaussian_defaults = {
        "min_std_ratio": t2_gaussian_mf_defaults["min_std_ratio"],
        "uncertainty_factor": t2_gaussian_mf_defaults["uncertainty_factor"],
        "decimal_places": t2_gaussian_mf_defaults["decimal_places"],
    }

    algorithm_col, parameter_col, value_col = [], [], []

    for alg, params in alg_defaults.items():
        alg_rows, param_rows, value_rows = expand_rows(
            f"{alg.upper()} Algorithm", params
        )
        algorithm_col.extend(alg_rows)
        parameter_col.extend(param_rows)
        value_col.extend(value_rows)

    # Add Gaussian MF section
    g_rows, p_rows, v_rows = expand_rows("Type-2 Gaussian MF", type_2_gaussian_defaults)
    algorithm_col.extend(g_rows)
    parameter_col.extend(p_rows)
    value_col.extend(v_rows)

    data = {"Algorithm": algorithm_col, "Parameter": parameter_col, "Value": value_col}
    return data
