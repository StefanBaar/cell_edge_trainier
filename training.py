import pytorch_lightning as pl

import argparse
import torch

from trainers import CellTrainer
from dataprep import load_train_val, MyDataset

import ssl


def get_args():
    parser = argparse.ArgumentParser(description='Produces training arguments:')
    parser.add_argument("--arch"        ,default="FPN"     ,type=str, nargs='?', const=1)
    parser.add_argument("--encoder_name",default="resnet18",type=str, nargs='?', const=1)
    parser.add_argument("--devices"     ,default=0         ,type=int, nargs='?', const=1)
    parser.add_argument("--batch_size"  ,default=35        ,type=int, nargs='?', const=1)
    parser.add_argument("--in_channels" ,default=1         ,type=int, nargs='?', const=1)
    parser.add_argument("--out_channels",default=4         ,type=int, nargs='?', const=1)
    parser.add_argument("--max_epochs"  ,default=1         ,type=int, nargs='?', const=1)
    return parser
    

def multi_class_train(main_path,arch,encoder_name,batch_size,in_channels,out_channels,devices,max_epochs):
    
    model_dic = main_path+"models/"
    log_dir   = main_path+"lightning_logs/"
    name      = arch+"_"+encoder_name+"_"+str(in_channels)+"_"+str(out_channels)+"_"+str(max_epochs)+"_"+str(batch_size)
    
    train_path = main_path+"training/"
    valid_path = main_path+"validation/"

    TRAIN_DATA = MyDataset(train_path)
    VALID_DATA = MyDataset(valid_path)

    train_data, valid_data = load_train_val(TRAIN_DATA,VALID_DATA,batch_size)

    model = CellTrainer(arch, encoder_name, in_channels, out_channels)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir+name+"/",flush_secs=10)
    if torch.cuda.is_available():
        trainer   = pl.Trainer(accelerator="gpu", devices=devices,max_epochs=max_epochs,
                            logger=tb_logger,)
    else:
        trainer   = pl.Trainer(accelerator="cpu", max_epochs=max_epochs,
                            logger=tb_logger,)

    trainer.fit(model, train_dataloaders=train_data,val_dataloaders=valid_data)
    
    torch.save(model,model_dic+arch+"_"+encoder_name+"_"+str(in_channels)+"_"+str(out_channels)+"_"+str(max_epochs)+"_"+str(batch_size))

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    main_path    = "./data/"

    
    #arch         = "FPN"
    #encoder_name = "resnet18"
    #devices      = [0]
    #batch_size   = 30
    #in_channels  = 1
    #out_channels = 3
    #max_epochs   = 1
    args         =  get_args().parse_args()
    arch         =  args.arch
    encoder_name =  args.encoder_name
    devices      = [args.devices]
    batch_size   =  args.batch_size
    in_channels  =  args.in_channels
    out_channels =  args.out_channels
    max_epochs   =  args.max_epochs
    
    print(args)


    multi_class_train(main_path,
                      arch,
                      encoder_name,
                      batch_size,
                      in_channels,
                      out_channels,
                      devices,
                      max_epochs)
    