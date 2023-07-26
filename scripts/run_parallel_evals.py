import datetime
import subprocess
import sys
from typing import List


def main(args, seeds: List[int] = [10, 123456, 54321, 123152, 19501]):
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ldir = "hydra.run.dir=${log_dir}/${eval_name}/" + timestr + "/${seed}"
    cmds = [
        [
            "python",
            "scripts/evaluate_ndf_mug_standalone.py",
            *args,
            f"seed={seed}",
            ldir,
        ]
        for seed in seeds
    ]
    processes = [subprocess.Popen(program) for program in cmds]

    # wait
    for process in processes:
        process.wait()


if __name__ == "__main__":
    # Join all command line args into a string

    main(sys.argv[1:])
