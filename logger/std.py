import logging
import os
import hydra
from omegaconf import OmegaConf


class StdLogger():
    def __init__(self, file_path):

        top_level = os.path.dirname(os.path.dirname(__file__))

        if not os.path.exists(os.path.join(top_level, "outputs")):
            os.mkdir(os.path.join(top_level, "outputs"))

        logging.basicConfig(filename=os.path.join(top_level, file_path),
                            level=logging.INFO,
                            format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        self.logger = logging.getLogger(__file__)
logger = StdLogger("outputs/std.log").logger
