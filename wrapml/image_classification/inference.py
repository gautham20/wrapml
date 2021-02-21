import torch
from tqdm.auto import tqdm

def predict(model, dataloader, index=''):
    model.eval()
    device = model.device
    with torch.no_grad():
        predictions = []
        for batch in tqdm(dataloader, leave=False, desc=f'test {index}'):
            batch = batch.to(device)
            predictions.append(
                torch.softmax(model(batch), 1)
            )
            del batch
    return torch.cat(predictions).detach().cpu()

def inference_with_tta(model, dataloader, n_tta=3, reduction='mean'):
    tta_predictions = []
    for i in range(n_tta):
        tta_predictions.append(predict(model, dataloader, i))
    if reduction == 'mean':
        return torch.mean(torch.stack(tta_predictions), dim=0)
    if reduction == 'max':
        predictions ,_ = torch.max(torch.stack([tta_predictions[0], tta_predictions[1]]), dim=0)
        return predictions
    return tta_predictions

def multi_model_inference_with_tta(models, dataloader, n_tta=3, reduction='mean'):
    tta_predictions = []
    batch_size = dataloader.batch_size
    n_classes = 5
    with torch.no_grad():
        for i in range(n_tta):
            predictions = np.zeros((len(dataloader.dataset), n_classes))
            for bi, batch in enumerate(tqdm(dataloader, leave=False, desc=f'test tta {i}')):
                for model in models:
                    model.eval()
                    device = model.device
                    batch = batch.to(device)
                    predictions[bi*batch_size: bi*batch_size + len(batch), :] += torch.softmax(model(batch), 1).detach().cpu().numpy() * 1/len(models)
            tta_predictions.append(predictions)
    if reduction == 'mean':
        return np.mean(tta_predictions, axis=0)
    if reduction == 'max':
        return np.max(tta_predictions, axis=0)
    return tta_predictions

def load_inference_models(model_cls, models_path, folds=None, checkpoint_selection='best'):
    models = {}
    for model_dir in models_path.glob('*'):
        model_name = str(model_dir).split('/')[-1]
        fold = int(model_name.split('-')[-1][4:])
        if folds is None or (fold in folds):
            model_ckpts = model_dir.glob('*')
            if checkpoint_selection == 'last':
                models[f'{model_name}_last'] = model_cls.load_from_checkpoint(
                    model_dir/'last.ckpt'
                )
                if torch.cuda.is_available():
                    models[f'{model_name}_last'].to('cuda')
            else:
                ckpt_names = [str(x).split('/')[-1] for x in model_ckpts]
                ckpt_names.remove('last.ckpt')
                ckpt_names = sorted(ckpt_names, key=lambda x: int(x.split('-')[0][5:]), reverse=True)
                models[f'{model_name}_{ckpt_names[0]}'] = model_cls.load_from_checkpoint(
                    model_dir/ckpt_names[0]
                )
                if torch.cuda.is_available():
                    models[f'{model_name}_{ckpt_names[0]}'].to('cuda')
    return models