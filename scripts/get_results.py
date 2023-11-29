import os

import pandas as pd
import typer

from taxpose.utils.compile_result import get_result_df


def main(res_dir: str, n_trials: int = 100):
    # Get the name of every directory in res_dir
    seeds = [
        seed
        for seed in os.listdir(res_dir)
        if os.path.isdir(os.path.join(res_dir, seed))
    ]

    result_file = f"trial_{n_trials-1}/success_rate_eval_implicit.npz"

    dfs = []
    for seed in seeds:
        # Get the path to the result file
        result_path = os.path.join(res_dir, seed, result_file)

        # res = np.load(result_path)
        # # Set the index of the dataframe to be the seed

        # succ = np.logical_and(
        #     res["grasp_success_list"], res["place_success_teleport_list"]
        # )
        # mp_succ = np.logical_and(res["grasp_success_list"], res["place_success_list"])

        # ress = {
        #     "seed": seed,
        #     "Grasp": res["grasp_success_list"].astype(float).mean(),
        #     "Place": res["place_success_teleport_list"].astype(float).mean(),
        #     "Overall": succ.astype(float).mean(),
        #     "Overall_mp": mp_succ.astype(float).mean(),
        # }

        # for thresh in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        #     ps_at_t = np.logical_and(
        #         res["place_success_teleport_list"], res["penetration_list"] < thresh
        #     ).mean(axis=-1)
        #     s_at_t = np.logical_and(succ, res["penetration_list"] < thresh).mean(
        #         axis=-1
        #     )

        #     ress[f"Place@{thresh}"] = ps_at_t.mean()
        #     ress[f"Overall@{thresh}"] = s_at_t.mean()

        # results = pd.DataFrame(
        #     ress,
        #     columns=["seed", "Grasp", "Place", "Overall", "Overall_mp"]
        #     + [f"Place@{thresh}" for thresh in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]]
        #     + [f"Overall@{thresh}" for thresh in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]],
        #     index=[0],
        # )
        # # set the seed column as the index
        # results.set_index("seed", inplace=True)
        results = get_result_df(result_path, seed)
        dfs.append(results)

    # Concatenate all the results
    df = pd.concat(dfs)

    # Save the results
    df.to_csv(os.path.join(res_dir, "results.csv"))
    print(df)
    return df


if __name__ == "__main__":
    typer.run(main)
