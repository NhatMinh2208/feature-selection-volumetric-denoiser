import torch
import random
import datagen
import model.losses as losses
import model.models as models
import utils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dropout_features(dictionary, p=0.5):
    if not isinstance(dictionary, dict):
        raise TypeError("Input must be a dictionary.")
    
    for key in dictionary:
        if random.random() < p:
            if(key == 'color'):
                continue
            size = dictionary[key].size()
            if(key == 'T_ss' or key == 'T_ms'):
                dictionary[key] = torch.ones(size)
                continue
            dictionary[key] = torch.zeros(size)
    return dictionary

features = ['Lss', 'albedo_ss', 'sigma_s_ss', 'sigma_t_ss', 'position_ss',
                         'T_ss', 'tau_ss', 'z_v_ss', 'L_ms', 'albedo_ms', 'sigma_s_ms',
                        'sigma_t_ms', 'position_ms', 'T_ms', 'tau_ms', 'z_v_ms', 'z_cam']
selector_dataset = VFODataset('./data/test')
def seed_fn(id): np.random.seed()
sdata_loader = torch.utils.data.DataLoader(selector_dataset, num_workers=0, pin_memory=True, batch_size=1, worker_init_fn=seed_fn)
model = torch.load('./models/dual_l1_vfo_probe.pt', map_location=device)
model = model.to(device)
#model.load_state_dict(checkpoint)

def dropout_features2(dictionary, features):
    if not isinstance(dictionary, dict):
        raise TypeError("Input must be a dictionary.")
    
    for key in dictionary:
        if(key == 'color'):
            continue
        if key in features:
            continue
        size = dictionary[key].size()
        if(key == 'T_ss' or key == 'T_ms'):
            dictionary[key] = torch.ones(size)
            continue
        dictionary[key] = torch.zeros(size)
    return dictionary

def mean_loss(features, device, dataloader, model, loss_fn):
    c = 0
    permutation = [0, 3, 1, 2]
    eval_loss = 0
    with torch.no_grad():
        for _data in dataloader:
            _data = dropout_features2(_data, features)
            _data = datagen.pre_process(_data)
            _data = to_torch_tensors(_data)
            data_noisy = _data['color'].permute(permutation).to(device)
            data_ss = _data['ss'].permute(permutation).to(device)
            data_ms = _data['ms'].permute(permutation).to(device)
            target = _data['GT'].permute(permutation).to(device)
            prediction =  model(data_noisy, data_ss, data_ms)
            eval_loss += loss_fn(prediction, target).item()
        eval_loss = eval_loss * (1/len(dataloader))
    return eval_loss

def features_selection(features: list[str], dataloader, model, loss_fn):
    feature_set = []
    feature_sets = [feature_set + []]
    loss = mean_loss(feature_set, device, dataloader, model, loss_fn)
    losses = [loss]
    remain_feature = features
    for i in range(len(features)):
        L = []
        best_loss = float('inf')
        for f in remain_feature:
            loss_prime = mean_loss(feature_set + [f], device, dataloader, model, loss_fn)
            L.append((loss_prime, f))
        best_loss = min(L, key = lambda t: t[0])
        feature_set.append(best_loss[1])
        feature_sets.append(feature_set + [])
        losses.append(best_loss[0])
        remain_feature.remove(best_loss[1])
    return feature_sets, losses



