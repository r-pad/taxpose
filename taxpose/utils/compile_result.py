import numpy as np
import pandas as pd


def get_result_df(result_path: str, seed: str):
    res = np.load(result_path)
    # Set the index of the dataframe to be the seed

    succ = np.logical_and(res["grasp_success_list"], res["place_success_teleport_list"])
    mp_succ = np.logical_and(res["grasp_success_list"], res["place_success_list"])

    ress = {
        "seed": seed,
        "Grasp": res["grasp_success_list"].astype(float).mean(),
        "Place": res["place_success_teleport_list"].astype(float).mean(),
        "Overall": succ.astype(float).mean(),
        "Overall_mp": mp_succ.astype(float).mean(),
    }

    for thresh in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        ps_at_t = np.logical_and(
            res["place_success_teleport_list"], res["penetration_list"] < thresh
        ).mean(axis=-1)
        s_at_t = np.logical_and(succ, res["penetration_list"] < thresh).mean(axis=-1)

        ress[f"Place@{thresh}"] = ps_at_t.mean()
        ress[f"Overall@{thresh}"] = s_at_t.mean()

    results = pd.DataFrame(
        ress,
        columns=["seed", "Grasp", "Place", "Overall", "Overall_mp"]
        + [f"Place@{thresh}" for thresh in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]]
        + [f"Overall@{thresh}" for thresh in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]],
        index=[0],
    )
    # set the seed column as the index
    results.set_index("seed", inplace=True)

    return results
