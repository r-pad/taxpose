import os

import numpy as np
import pandas as pd
import typer


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

        res = np.load(result_path)
        # Set the index of the dataframe to be the seed

        succ = np.logical_and(
            res["grasp_success_list"], res["place_success_teleport_list"]
        )
        results = pd.DataFrame(
            {
                "seed": seed,
                "Grasp": res["grasp_success_list"].astype(float).mean(),
                "Place": res["place_success_teleport_list"].astype(float).mean(),
                "Overall": succ.astype(float).mean(),
            },
            columns=["seed", "Grasp", "Place", "Overall"],
            index=[0],
        )
        # set the seed column as the index
        results.set_index("seed", inplace=True)
        dfs.append(results)

    # Concatenate all the results
    df = pd.concat(dfs)

    # Save the results
    df.to_csv(os.path.join(res_dir, "results.csv"))


if __name__ == "__main__":
    typer.run(main)
