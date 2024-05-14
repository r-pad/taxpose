import time

import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import FS10_V1

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
)
env = Environment(action_mode, headless=True)
env.launch()

train_tasks = FS10_V1["train"]
test_tasks = FS10_V1["test"]
task_to_train = np.random.choice(train_tasks, 1)[0]
task = env.get_task(task_to_train)
task.sample_variation()  # random variation
descriptions, obs = task.reset()

# Step for 10s wall clock time
start = time.time()
while time.time() - start < 10:
    obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
