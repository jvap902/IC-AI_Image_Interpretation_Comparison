from torch_cka import CKA
import torch

def cka(fst_modelc, snd_modelc):

    cka = CKA(fst_modelc.model, snd_modelc.model,
              model1_name=fst_modelc.name, model2_name=snd_modelc.name,
              #model1_layers=fst_modelc.model.named_modules, model2_layers=snd_modelc.model.named_modules, a princípio extrai de todos por padrão
              device='cuda' if torch.cuda.is_available() else 'cpu')
    
    cka.compare(dataloader1=fst_modelc.val_loader, dataloader2=snd_modelc.val_loader)

    cka.plot_results()

    return