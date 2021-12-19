#!/usr/bin/env python3
import imageio
import numpy
from tqdm import tqdm
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from gameEnv import GameEnv
import numpy as np
import sys


def main():
    dataPath = "dinodqn"
    #This is for concurrent run-up
    env = SubprocVecEnv([lambda: GameEnv(96, 96) for i in range(4)])
    train = None
    try:
        train = sys.argv[1]
    except IndexError:
        train = None
        print("Select train mode: true for training, false for results")
        sys.exit(1)

    # Training Mode from Stable Baselines
    if train.lower() == "true":
        callback_data = CheckpointCallback(
            save_freq=4000, save_path="./check", name_prefix=dataPath
        )
        model = PPO2(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./tensor_dino",
        )
        model.learn(total_timesteps=250000, callback=[callback_data])
        model.save(dataPath)
    # Load Module
    model = PPO2.load(dataPath, env=env)
    images = []
    observe = env.reset()
    renderImage = model.env.render(mode="rgb_array")
    for _ in tqdm(range(400)):
        images.append(renderImage)
        action, _ = model.predict(observe, deterministic=True)
        observe, _, _, _  = env.step(action)
        renderImage = env.render(mode="rgb_array")
    imageio.mimsave(
        "game.gif", [numpy.array(renderImage) for i, renderImage in enumerate(images)], fps=15
    )
    exit()


if __name__ == "__main__":
    main()
