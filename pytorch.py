import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

#读取西游记
lines=open("xiyouji.txt",'r',encoding='utf-8').read()


