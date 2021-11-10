import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ProgressiveImageResizing(Callback):
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        epoch_final_size: int,
        n_step: int,
        init_size: int,
        final_size: int,
    ):
        super().__init__()
        self.data_module = data_module

        step = epoch_final_size // n_step
        self.epoch_to_change = np.linspace(
            1, epoch_final_size, n_step, dtype=int
        ).tolist()
        self.new_size = np.linspace(init_size, final_size, n_step, dtype=int).tolist()

        self.epoch = 1

        self._change_size(self.new_size[0])

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self.epoch += 1

        if self.epoch in self.epoch_to_change:
            idx = self.epoch_to_change.index(self.epoch)
            self._change_size(self.new_size[idx])
            self.log("size", self.new_size[idx])

    def _change_size(self, size: int):
        self.data_module.train_dataset.dataset.transform = get_transforms((size, size))
