from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import frame_conversion
from tudatpy import util,math
import numpy as np

def format_residual_history(residual_history, obs_times, state_history):
    residuals_per_iteration = []
    rsw_residuals_per_iteration = []

    for i in range(residual_history.shape[1]):
        res_i = residual_history[:, i]
        reshaped_residuals = res_i.reshape(-1, 3)
        residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), reshaped_residuals]))

        rsw_residuals = rotate_inertial_3_to_rsw(np.array(obs_times).reshape(-1, 1),
                                                 reshaped_residuals, state_history)

        rsw_residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), rsw_residuals]))

    return residuals_per_iteration, rsw_residuals_per_iteration



def make_rsw_rotation_from_state_history(state_history, size=3):

    if type(state_history) == dict:
        state_history_dict = state_history
    else:
        state_history_dict = array_to_dict(state_history)

    state_history_interpolator = math.interpolators.create_one_dimensional_vector_interpolator(
        state_history_dict, math.interpolators.lagrange_interpolation(8))

    def rsw_rotation_at_epoch(sample_epoch):

        R = frame_conversion.inertial_to_rsw_rotation_matrix(state_history_interpolator.interpolate(sample_epoch))

        if size == 3:
            pass

        elif size == 6:
            R = linalg.block_diag(R, R)

        else:
            raise NotImplementedError("Stacking of rotation matrices over size 6 not implemented.")

        return R

    return rsw_rotation_at_epoch



def rotate_inertial_3_to_rsw(epochs, inertial_3, state_history):

    R_func = make_rsw_rotation_from_state_history(state_history, size=3)
    rsw_coll = []
    for epoch, inertial in zip(epochs, inertial_3):

        R = R_func(epoch)
        rsw = R.dot(inertial)
        rsw_coll.append(rsw)


    assert np.array(rsw_coll).shape == inertial_3.shape

    return np.array(rsw_coll)


def rotate_inertial_6_to_rsw(epochs, inertial_6, state_history):

    R_func = make_rsw_rotation_from_state_history(state_history, size=6)
    rsw_coll = []

    for epoch, inertial in zip(epochs, inertial_6):

        R = R_func(epoch)
        rsw = R.dot(inertial)
        rsw_coll.append(rsw)

    assert np.array(rsw_coll).shape == inertial_6.shape

    return np.array(rsw_coll)



def array_to_dict(state_history_array):

    state_history_dict = dict()

    for row in state_history_array:

        state_history_dict[row[0]] = row[1:7]

    return state_history_dict
