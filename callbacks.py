import pytorch_lightning as pl

class AvgLossCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        avg_loss = logs['loss']
        trainer.logger.log_metrics({'avg_train_loss':avg_loss}, step=trainer.callback_metrics['epoch'])

