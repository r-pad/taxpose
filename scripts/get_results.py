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
