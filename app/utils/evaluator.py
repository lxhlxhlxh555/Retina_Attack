import abc
import torch.nn.functional as F
class Evaluator(object):
    @abc.abstractmethod
    def evaluate(self,model,dataloader,device):
        pass

class ClassificationEvaluator(Evaluator):
    def evaluate(self,model,dataloader,device):
        correct = 0
        all = 0
        for i, batch in enumerate(dataloader):
            Xs = batch['image']
            gts = batch['label']
            Xs = Xs.to(device)
            gts = gts.to(device)
            all += gts.shape[0]
            # print(Xs)
            # print(gts)
            preds = model(Xs)
            preds = F.softmax(preds,dim=-1)
            print(preds)
            preds = preds.max(dim=-1)[1]
            print(gts)
            print(preds)
            correct += (gts == preds).sum()
        acc = (correct / all).item()
        return acc

def build_evaluator(params):
    task_type = params['task']
    if task_type == 'classification':
        evaluator = ClassificationEvaluator()
    return evaluator