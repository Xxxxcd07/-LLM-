from scripts import run_cross_schedule
from src.features import correction_features, fsl_light_features


def test_cross_schedule_method_specs_bind_corrections_and_feature_functions():
    ridge_obj = object()
    fsl_obj = object()
    corrections = {"ridge": ridge_obj, "fsl_light": fsl_obj}

    methods = {
        method: (correction_name, correction, feature_fn)
        for method, correction_name, correction, feature_fn in run_cross_schedule.method_specs("tissue", corrections)
    }

    assert methods["tissue"] == ("none", None, None)
    assert methods["tissue_plus_ridge"] == ("ridge", ridge_obj, correction_features)
    assert methods["tissue_plus_fsl_light"] == ("fsl_light", fsl_obj, fsl_light_features)


def test_cross_schedule_correction_feature_fns_include_fsl_light():
    configs = run_cross_schedule.correction_configs()

    assert configs["fsl_light"].correction == "fsl_light"
    assert run_cross_schedule.method_specs("mpl", object())[2][0] == "mpl_plus_fsl_light"
