import yaml
from addict import Dict
from pytorch_lightning import loggers
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def read_yaml(fpath=None):
    with open(fpath, mode='r') as file:
        yam = yaml.load(file, Loader=yaml.Loader)
        return Dict(yam)

def load_loggers(cfg):
    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = Path(cfg.config).parent
    version_name = Path(cfg.config).name[:-5]
    cfg.log_path = Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'
    print(f'---->Log dir: {cfg.log_path}')
    tb_logger = loggers.TensorBoardLogger(save_dir=log_path+str(log_name),
    name=version_name, version=f'fold{cfg.Data.fold}'
    , log_graph=True, default_hp_metric=False)
    csv_logger = loggers.CSVLogger(log_path+str(log_name), name = version_name, version = f'fold{cfg.Data.fold}')

    return [tb_logger, csv_logger]

def load_callbacks(cfg):
    callbacks = []

    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    if cfg.General.server == 'train' :
        callbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = True))
    return callbacks
