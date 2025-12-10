
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy
from sklearn.metrics import precision_recall_curve, confusion_matrix

@torch.no_grad()
def contrastive_evaluate(val_loader, model, ts_repository):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        ts_org = batch['ts_org'] #.cuda(non_blocking=True)
        target = batch['target'] #.cuda(non_blocking=True)
        ts_org = torch.from_numpy(ts_org).float()
        #ts_org=torch.unsqueeze(ts_org, dim=1)
        b, w, h = ts_org.shape
        target = torch.from_numpy(target)
        output = model(ts_org.view(b, h, w))

        output = ts_repository.weighted_knn(output)

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), ts_org.size(0))

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False, is_training=False):
    # Make predictions on a dataset with neighbors
    global features, nneighbors, fneighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)) #.cuda()

    if isinstance(dataloader.dataset, NeighborsDataset): # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        nneighbors = []
        fneighbors = []

    else:
        key_ = 'ts_org'
        include_neighbors = False

    ptr = 0
    features = None
    for batch in dataloader:
        ts = batch[key_]
        #ts = torch.unsqueeze(ts, dim=1)
        if ts.ndim == 3:
            bs, w, h = ts.shape
        else:
            bs, w = ts.shape
            h =1

        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).float()
            targets.append(torch.from_numpy(batch['target']))
        else:
            targets.append(batch['target'])

        res = model(ts.view(bs, h, w), forward_pass='return_all')
        output = res['output']

        if return_features:
            # Lazily allocate features with the true dimensionality
            if features is None:
                num_samples = len(dataloader.dataset)  # or loader.dataset depending on your variable name
                feat_dim = res['features'].size(1)
                features = torch.zeros(
                    num_samples,
                    feat_dim,
                    device=res['features'].device,
                    dtype=res['features'].dtype,
                )
            features[ptr: ptr+bs] = res['features']
            ptr += bs
        
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))

        if include_neighbors:
            nneighbors.append(batch['possible_nneighbors'])
            fneighbors.append(batch['possible_fneighbors'])

    predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        nneighbors = torch.cat(nneighbors, dim=0)
        fneighbors = torch.cat(fneighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': nneighbors, 'fneighbors': fneighbors} for pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]

    if return_features:
        feat_np = features.cpu().numpy()  # save features in csv
        fhdr = [str(x) for x in range(feat_np.shape[1])] + ['Class']
        # feat_np = np.hstack((feat_np, np.array(targets)[np.newaxis].T)) CUDA
        feat_np = np.hstack((feat_np, np.array(targets.cpu().numpy())[np.newaxis].T)) 

        feat_df = pd.DataFrame(feat_np, columns=fhdr)

        prob_np = np.array(out[0]['probabilities'])
        phdr = [str(x) for x in range(prob_np.shape[1])] + ['Class']
        # prob_np = np.hstack((prob_np, np.array(targets)[np.newaxis].T))
        prob_np = np.hstack((prob_np, np.array(targets.cpu().numpy())[np.newaxis].T)) 
        prob_df = pd.DataFrame(prob_np, columns=phdr)

        if is_training:
            feat_df.to_csv(p['classification_trainfeatures'], index=False, header=True, sep=',')
            prob_df.to_csv(p['classification_trainprobs'], index=False, header=True, sep=',')
        else:
            feat_df.to_csv(p['classification_testfeatures'], index=False, header=True, sep=',')
            prob_df.to_csv(p['classification_testprobs'], index=False, header=True, sep=',')

        return out, features.cpu()

    else:
        # tmp = np.array(out[0]['probabilities'])
        return out


@torch.no_grad()
def classification_evaluate(predictions):
    # Evaluate model based on classification loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        fneighbors = head['fneighbors']
        org_anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = org_anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()

        similarity = torch.matmul(probs, probs.t())
        fneighbors = fneighbors.contiguous().view(-1)
        anchors = org_anchors.contiguous().view(-1)
        similarity = similarity[anchors, fneighbors]
        ones = torch.ones_like(similarity)
        inconsistency_loss = F.binary_cross_entropy(similarity, ones).item()

        # Total loss
        total_loss = 5*entropy_loss + consistency_loss - 0*inconsistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'inconsistency': inconsistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'classification': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}

@torch.no_grad()
def pr_evaluate(all_predictions, class_names=None,
                compute_purity=True, compute_confusion_matrix=True,
                confusion_matrix_file=None, majority_label=0):

    head = all_predictions[0]
    targets = head['targets']              # torch tensor
    predictions = head['predictions']      # torch tensor
    probs = head['probabilities']          # torch tensor or np

    # Labels as ints on CPU
    labels = np.asarray(targets.cpu().numpy(), dtype=int)

    probs_np = probs.detach().cpu().numpy() if hasattr(probs, 'detach') else np.asarray(probs)
    # If these are logits (rows not summing ~1), softmax them safely.
    row_sums = probs_np.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-3) or np.any(probs_np < 0) or np.any(probs_np > 1):
        x = probs_np - probs_np.max(axis=1, keepdims=True)  # Numerically stable softmax
        ex = np.exp(x)
        probs_np = ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)

    # Preferred anomaly score: low confidence in ANY head → anomalous
    # This avoids relying on "majority_label" being the normal head.
    scores = 1.0 - probs_np.max(axis=1)

    # Sanitize and polarity auto-check
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # If constant scores -> AP is just positive rate; pick +inf threshold so nothing is flagged
    if np.allclose(np.std(scores), 0.0):
        ap = float(np.mean(labels))
        precision_at_f1 = float(ap) if ap > 0 else 0.0
        recall_at_f1    = 0.0 if ap == 0 else 1.0
        rep_f1          = 0.0
        metrics = {
            'precision':      precision_at_f1,
            'recall':         recall_at_f1,
            'f1_best':        float(rep_f1),
            'aupr':           float(ap),
            'best_threshold': float('inf'),
        }
        if class_names == 'Anom':
            anomalies = [0]*len(scores)  # no positives at +inf
            best_tn, best_fp, best_fn, best_tp = confusion_matrix(labels, anomalies).ravel()
            print("Anomalies --> TP:", best_tp, ", TN:", best_tn, ", FN:", best_fn, ", FP:", best_fp)
        return metrics

    # Polarity: positives should have higher scores than negatives
    pos_mean = float(np.mean(scores[labels == 1])) if np.any(labels == 1) else 0.0
    neg_mean = float(np.mean(scores[labels == 0])) if np.any(labels == 0) else 0.0
    if pos_mean <= neg_mean:
        scores = -scores   # flip so higher = more anomalous

    # PR / AUPR, using a valid threshold index (avoid index 0)
    from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)
    f1_score = np.nan_to_num(f1_score)

    # precision/recall have len = len(thresholds) + 1; index 0 has no threshold.
    if len(thresholds) > 0:
        best_idx = int(np.nanargmax(f1_score[1:]) + 1)  # force a valid threshold-backed index
        best_threshold = float(thresholds[best_idx - 1])
    else:
        # degenerate but finite case: no thresholds returned → fall back safely
        best_idx = int(np.nanargmax(f1_score))
        best_threshold = float('inf')

    rep_f1 = float(f1_score[best_idx])
    precision_at_f1 = float(precision[best_idx])
    recall_at_f1    = float(recall[best_idx])
    aupr            = float(average_precision_score(labels, scores))

    metrics = {
        'precision':      precision_at_f1,
        'recall':         recall_at_f1,
        'f1_best':        rep_f1,
        'aupr':           aupr,
        'best_threshold': best_threshold,
    }

    if class_names == 'Anom':
        anomalies = [1 if s >= best_threshold else 0 for s in scores]
        best_tn, best_fp, best_fn, best_tp = confusion_matrix(labels, anomalies).ravel()
        print("Anomalies --> TP:", best_tp, ", TN:", best_tn, ", FN:", best_fn, ", FP:", best_fp)

    return metrics

"""
@torch.no_grad()
def pr_evaluate(all_predictions, class_names=None,
                compute_purity=True, compute_confusion_matrix=True,
                confusion_matrix_file=None, majority_label=0):

    head = all_predictions[0]
    targets = head['targets'] #.cuda()
    predictions = head['predictions'] #.cuda()
    probs = head['probabilities'] #.cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    scores = 1-np.array(probs)[:,majority_label]
    # labels = np.array(targets).tolist() CUDA
    labels = np.array(targets.cpu().numpy()).tolist()

    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    try:
        f1_score = 2*precision*recall / (precision+recall)
        if np.isnan(f1_score).any():
            f1_score = np.nan_to_num(f1_score)
            print('f1: Nan --> 0')
    except ZeroDivisionError:
        f1_score = [0.0]
        print('f1: 0 --> 0')

    best_f1_index = np.argmax(f1_score)
    rep_f1 = f1_score[best_f1_index]   

    # === ADDED (new metrics, minimal change) ===
    from sklearn.metrics import average_precision_score
    aupr = float(average_precision_score(labels, scores))
    precision_at_f1 = float(precision[best_f1_index])
    recall_at_f1    = float(recall[best_f1_index])
    best_threshold  = float(thresholds[best_f1_index - 1]) if best_f1_index > 0 else float('-inf')

    metrics = {
        'precision':       precision_at_f1,
        'recall':          recall_at_f1,
        'f1_best':         float(rep_f1),
        'aupr':            aupr,
        'best_threshold':  best_threshold,
    }

    if class_names=='Anom':
        best_threshold = thresholds[best_f1_index]
        anomalies = [1 if s >= best_threshold else 0 for s in scores]
        best_tn, best_fp, best_fn, best_tp = confusion_matrix(labels, anomalies).ravel()
        print("Anomalies --> TP: ", best_tp, ", TN: ", best_tn, ", FN: ", best_fn, ", FP: ", best_fp)
        print(majority_label)
        print(metrics.classification_report(labels, anomalies))

    return metrics
"""

def replace_majority_label(flat_preds, majority_label):
    #unique_labels = torch.unique(flat_preds)
    new_pred = torch.where(flat_preds == majority_label, 0, 1)
    return new_pred