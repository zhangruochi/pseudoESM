import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logger.std import logger as std_logger

from abc import ABC

# @abc.abstractmethod

class Callback(ABC):

    def on_init_start(self):
        pass

    def on_init_end(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_evaluate_train_batch_end(self):
        pass

    def on_evaluate_valid_batch_end(self):
        pass

    def on_evaluate_test_batch_end(self):
        pass

    def on_epoch_end(self):
        pass


class MyPrintingCallback(Callback):

    def on_init_start(self):
        std_logger.info("Starting to init trainer!")

    def on_init_end(self, trainer):
        std_logger.info("trainer is init now")

    def on_train_batch_end(self, trainer):
        std_logger.info(
            'Train | Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f}'
            .format(trainer.epoch + 1, trainer.num_epoch,
                    trainer.train_batch_idx, trainer.total_train_batch,
                    trainer.train_loss.item()))

    def on_evaluate_train_batch_end(self, trainer):
        std_logger.info(
            'Evaluate | Train | Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f}'
            .format(trainer.epoch + 1, trainer.num_epoch,
                    trainer.train_batch_idx, trainer.train_evaluation_batch,
                    trainer.evaluation_train_batch_loss))

    def on_evaluate_valid_batch_end(self, trainer):
        std_logger.info(
            'Evaluate | Valid | Epoch: {0:03d}/{1:03d} | Batch {2:03d}/{3:03d} | Loss: {4:.3f}'
            .format(trainer.epoch + 1, trainer.num_epoch,
                    trainer.valid_batch_idx, trainer.valid_evaluation_batch,
                    trainer.valid_batch_loss))

    def on_epoch_end(self, trainer):
        pass