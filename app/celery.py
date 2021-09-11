from celery import Celery
from flask import current_app
import torch
from torch.utils.data import DataLoader
from .utils.attacker import build_attacker
from .utils.evaluator import build_evaluator
from .utils.dataset import build_dataset
from .utils.attack_tools import load_model
celery_app = Celery(__name__)

@celery_app.task(bind=True)
def model_test(self,params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(params['model'],device)
    dataset = build_dataset(params)
    evaluator = build_evaluator(params)
    attacker = build_attacker(params)
    dataloader = DataLoader(dataset)
    orig_acc = evaluator.evaluate(model,dataloader,device)
    new_dataset = attacker.run(model,dataset,device)
    new_loader = DataLoader(new_dataset)
    new_acc = evaluator.evaluate(model,new_loader,device)
    print("Orig_acc:",orig_acc)
    print("New_acc:",new_acc)
    return {'code':200, 'dataset_size':len(dataset), 'orig_acc':orig_acc, 'new_acc':new_acc,'status':'Task Complete!'}