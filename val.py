from pathlib import Path
import sys
import os
from datetime import datetime
from model.utils.inferScr.infer import val
#val script for validation of the model, it will generate a log file in the specified path with the results of the validation.
if __name__ == "__main__":
    val(path="model/res/val")