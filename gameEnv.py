import cv2
import base64
import gym
from gym import spaces
import numpy
import os 
import time
from collections import deque
from io import BytesIO
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait



class GameEnv(gym.Env):

    def __init__(self,sheight:int, swidth:int) -> None:
        
        #Screen definitions
        self.height = sheight
        self.width = swidth
        # Actions are 3: jump, duck and no action
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0,high=255,shape=(self.width, self.height,4), dtype=numpy.uint)
        # Headstart Queue
        self.__key = None
        self.gameQ = deque(maxlen=4)
        #Chrome and Webdriver
        self.__options = webdriver.ChromeOptions
        self.__options.add_argument("--mute-audio")
        self.__driver = webdriver.Chrome(ChromeDriverManager.install(), self.__options)
        daction_chains = ActionChains(self.__driver)
        self.pressAction = {
            "up": daction_chains.key_down(Keys.ARROW_UP),
            "down": daction_chains.key_down(Keys.ARROW_DOWN),
            "nothing": daction_chains.key_down(Keys.ARROW_RIGHT)
        }
        self.releaseAction = {
            "up": daction_chains.key_up(Keys.ARROW_UP),
            "down": daction_chains.key_up(Keys.ARROW_DOWN),
            "nothing": daction_chains.key_up(Keys.ARROW_RIGHT)
        }
        super().__init__()