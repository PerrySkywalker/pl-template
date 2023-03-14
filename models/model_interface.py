import pytorch_lightning as pl
import torch

class ModelInterface(pl.LightningModule):

    def __init__(self, model, **kargs):
        super(ModelInterface, self).__init__()
        



    def training_step(self, batch, batch_idx):

        return 

    def training_epoch_end(self, training_step_outputs):
        return 

    
    def validation_step(self,):
        return 


    def validation_step_end(self,):
        return 


    def test_step(self,):
        return 


    def test_step_end(self,):
        return 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return 