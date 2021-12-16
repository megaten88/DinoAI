import cv2
import base64
import gym
from gym import spaces
import numpy
import os
import time
from collections import deque
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.remote.webelement import WebElement


class GameEnv(gym.Env):
    def __init__(self, sheight: int, swidth: int) -> None:

        # Screen definitions
        self.height = sheight
        self.width = swidth
        # Actions are         self.__driver.get("chrome://dino")3: jump, duck and no action
        self.canvas = (By.CLASS_NAME, "runner-canvas")
        self.body = (By.TAG_NAME, "body")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.width, self.height, 4), dtype=numpy.uint
        )
        # Headstart Queue
        self.__key = None
        self.gameQ:deque = deque(maxlen=4)
        # Chrome and Webdriver
        self.__options = webdriver.ChromeOptions
        self.__options.add_argument("--mute-audio")
        self.__driver = webdriver.Chrome(ChromeDriverManager.install(), self.__options)
        # Actions in Dictionaries, to make calls easier
        self.pressAction = {
            "up":Keys.ARROW_UP,
            "down":Keys.ARROW_DOWN,
            "nothing":Keys.ARROW_RIGHT,
        }
        super().__init__()

    def imageBase64(self) -> numpy.array:
        #Getting image to canvas and transfrom to numpy array from base64
        imageFromCanvas = self.__driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL().substring(22)"
        )
        return numpy.array(Image.open(BytesIO(base64.b64decode(imageFromCanvas))))

    def getGameScore(self) -> int:
        '''Score can actually be taken from JavaScript scripts on the game
            What we do is take the core array and parse it into an INT
        '''
        getScore:str = ''.join(self.__driver.execute_script("return Runner.instance_.distanceMeter.digits"))
        return int(getScore)

    def gameObservation(self) -> numpy.stack:
        getImage = cv2.cvtColor(self.imageBase64, cv2.COLOR_BGR2GRAY)
        image = getImage[:500, :480]
        image = cv2.resize(image, (self.width, self.height))
        self.gameQ.append(image)

        if len(self.gameQ) < 4:
            return numpy.stack([image] * 4, axis=-1)
        else:
            return numpy.stack(self.gameQ, axis=-1)
    
    def isPlaying(self)-> bool:
            return self._driver.execute_script("return Runner.instance_.playing")

    ## Method recommended according to the mode gym
    def reset(self) -> numpy.stack:
        self.__driver.get('chrome://dino')
        WebDriverWait(self.__driver, 10).until(
            EC.presence_of_element_located(self.canvas)
        )
        self.__driver.find_element(*self.body).send_keys(Keys.SPACE)
        return self.gameObservation()

    def step(self, action:str):
        bodyElement:WebElement = self.__driver.find_element(*self.body)
        bodyElement.send_keys(self.pressAction[action])
        observe:numpy.stack = self.gameObservation()
        isPlaying:bool = self.isPlaying()
        reward = .1 if isPlaying else -1
        time.sleep(.01)
        return observe, reward, isPlaying, {"score": self.getGameScore()}

    def render(self, mode: str='human'):
        img = cv2.cvtColor(self.imageBase64, cv2.COLOR_BGR2RGB)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        


