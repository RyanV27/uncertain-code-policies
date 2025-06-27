# !pip install numpy scipy shapely astunparse pygments > /dev/null 2>&1
# !pip install imageio==2.4.1 imageio-ffmpeg pybullet moviepy

# To ensure that moviepy package does not try to reinstall ffmpeg 
# when you import moviepy, in /miniconda3/envs/{YOUR_ENV_NAME}/lib/
# python3.8/site-packages/moviepy/editor.py file, comment out 
# line 26; imageio.plugins.ffmpeg.download() and add a "pass" under it.

import os
import pybullet
import pybullet_data
import numpy as np
import threading
import copy
import cv2
from moviepy.editor import ImageSequenceClip
from huggingface_hub import InferenceClient
import subprocess

# imports for LMPs
import shapely
import ast
import astunparse
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

# Importing user-defined fucntions and classes
from lmp import FunctionParser, var_exists, merge_dicts, exec_safe 
from tabletop_sim_env import Robotiq2F85, PickPlaceEnv
from lmp_utils import setup_LMP

from tabletop_config import ALL_BLOCKS, ALL_BOWLS, cfg_tabletop, model_name


# Hugging face model and access token
HG_TOKEN = "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"

# Defining an InferenceClient from Hugging Face
client = InferenceClient(
    provider="auto",
    api_key=HG_TOKEN,
)


def main():
    print("Running Interactive_Demo.py!\n")

    # Download PyBullet assets.
    if not os.path.exists('ur5e/ur5e.urdf'):
        print('ur5e/ur5e.urdf doesn\'t exist')
        subprocess.run(['gdown', '--id', '1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc'], check=True)
        subprocess.run(['unzip', 'ur5e.zip'], check=True)

    if not os.path.exists('robotiq_2f_85/robotiq_2f_85.urdf'):  
        print('robotiq_2f_85/robotiq_2f_85.urdf doesn\'t exist')
        subprocess.run(['gdown', '--id', '1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX'], check=True)
        subprocess.run(['unzip', 'robotiq_2f_85.zip'], check=True)

    if not os.path.exists('bowl/bowl.urdf'):
        print('bowl/bowl.urdf doesn\'t exist')  
        subprocess.run(['gdown', '--id', '1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM'], check=True)
        subprocess.run(['unzip', 'bowl.zip'], check=True)


    #  Initialize the Environment
    num_blocks = 3 #@param {type:"slider", min:0, max:4, step:1}
    num_bowls = 3 #@param {type:"slider", min:0, max:4, step:1}
    high_resolution = False #@param {type:"boolean"}
    high_frame_rate = False #@param {type:"boolean"}

    # setup env and LMP
    env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
    block_list = np.random.choice(ALL_BLOCKS, size=num_blocks, replace=False).tolist()
    bowl_list = np.random.choice(ALL_BOWLS, size=num_bowls, replace=False).tolist()
    obj_list = block_list + bowl_list
    _ = env.reset(obj_list)
    lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, model_name=model_name, inference_client=client)

    # display env
    # cv2.imshow('env', cv2.cvtColor(env.get_camera_image(env), cv2.COLOR_BGR2RGB))

    print('\nAvailable objects:')
    print(obj_list)

    # Interactive Demo
    while True:
        user_input = input("User input: ") #@param {allow-input: true, type:"string"}

        env.cache_video = []

        print('Running policy and recording video...')
        lmp_tabletop_ui(user_input, f'objects = {env.object_list}')

    # render video
    # if env.cache_video:
    #     rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
    #     # display(rendered_clip.ipython_display(autoplay=1, loop=1))
    #     rendered_clip.write_videofile("output_video.mp4", codec="libx264")

if __name__ == "__main__":
    main()