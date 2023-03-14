import pytorch_lightning as pl
import argparse
from utils.utils import *
from datasets import DataInterface
from models import ModelInterface
from pytorch_lightning import Trainer
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='yamls/CD.yaml', type=str)
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--fold', default=0)
    args = parser.parse_args()
    return args



def main(cfg):

    #seed
    pl.seed_everything(cfg.General.seed)
    cfg.load_loggers = load_loggers(cfg)
    cfg.callbacks = load_callbacks(cfg)

    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}

    dm = DataInterface(**DataInterface_dict)

    ModelInterface_dict = {'model': cfg.Model,
                        'loss': cfg.Loss,
                        'optimizer': cfg.Optimizer,
                        'data': cfg.Data,
                        'log': cfg.log_path
                        }

    model = ModelInterface(**ModelInterface_dict)
    
    trainer = Trainer(
        #num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_backend = 'apex',
        amp_level=cfg.General.amp_level,
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths]
        #model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        path = model_paths[0]
        print(path+"-----------------------------------------------------------------------------")
        new_model = model.load_from_checkpoint(checkpoint_path='logs/Camelyon/TransMIL/fold0/last-v8.ckpt', cfg=cfg)
        trainer.test(model=new_model, datamodule=dm)


if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    main(cfg)