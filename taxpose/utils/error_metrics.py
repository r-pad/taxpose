import torch
from pytorch3d.transforms import Transform3d, Rotate, Translate
from taxpose.utils.se3 import get_degree_angle, get_translation

def get_2rack_errors(pred_T_action, T0, T1, mode="batch_min_rack", verbose=False):
    """
    Debugging function for calculating the error in predicting a mug-on-rack transform
    when there are 2 possible racks to place the mug on, and they are 0.3m apart in the x direction
    """
    assert mode in ["demo_rack", "bad_min_rack", "batch_min_rack"]

    if mode == "demo_rack":
        error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))

        error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))
    elif mode == "bad_min_rack":
        error_R_max0, error_R_min0, error_R_mean0 = get_degree_angle(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))
        error_t_max0, error_t_min0, error_t_mean0 = get_translation(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))
        error_R_max1, error_R_min1, error_R_mean1 = get_degree_angle(T0.inverse().translate(0.3, 0, 0).compose(
            T1).compose(pred_T_action.inverse()))
        error_R_max2, error_R_min2, error_R_mean2 = get_degree_angle(T0.inverse().translate(-0.3, 0, 0).compose(
            T1).compose(pred_T_action.inverse()))
        error_t_max1, error_t_min1, error_t_mean1 = get_translation(T0.inverse().translate(0.3, 0, 0).compose(
            T1.compose(pred_T_action.inverse())))
        error_t_max2, error_t_min2, error_t_mean2 = get_translation(T0.inverse().translate(-0.3, 0, 0).compose(
            T1.compose(pred_T_action.inverse())))
        error_R_mean = min(error_R_mean0, error_R_mean1, error_R_mean2)
        error_t_mean = min(error_t_mean0, error_t_mean1, error_t_mean2)
        if verbose:
            print(f'\t\/ a min over {error_t_mean0:.3f}, {error_t_mean1:.3f}, {error_t_mean2:.3f}')
    elif mode == "batch_min_rack":
        T = T0.inverse().compose(T1).compose(pred_T_action.inverse())
        Ts = torch.stack([
            T0.inverse().compose(T1).compose(pred_T_action.inverse()).get_matrix(),
            T0.inverse().translate(0.3, 0, 0).compose(T1).compose(pred_T_action.inverse()).get_matrix(),
            T0.inverse().translate(-0.3, 0, 0).compose(T1).compose(pred_T_action.inverse()).get_matrix(),
        ])

        error_R_mean, error_t_mean = 0, 0
        B = T0.get_matrix().shape[0]
        if verbose:
            print('\t\/ an average over ', end="")
        for b in range(B):  # for every batch
            _max, error_R_min, _mean = get_degree_angle(Transform3d(matrix=Ts[:, b, :, :]))
            error_R_mean += error_R_min

            _max, error_t_min, _mean = get_translation(Transform3d(matrix=Ts[:, b, :, :]))
            error_t_mean += error_t_min
            if verbose:
                print(f"{error_t_min:.3f}", end=" ")
        if verbose:
            print()
        error_R_mean /= B
        error_t_mean /= B
    else:
        raise ValueError("Invalid rack error type!")

    return error_R_mean, error_t_mean


def print_rack_errors(name, error_R_mean, error_t_mean):
    print(f"{name}- R error: {error_R_mean:.3f}, t error: {error_t_mean:.3f}")