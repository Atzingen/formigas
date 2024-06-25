import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("djay-de-gier-fopbf").project("ant-object-detection")
version = project.version(1)
dataset = version.download("yolov9")


project = rf.workspace("slowmo-games-ilya4").project("test-project-cubkg")
version = project.version(1)
dataset = version.download("yolov9")
