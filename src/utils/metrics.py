from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve

def compute_classification_metrics(y_true, y_prob, threshold=0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
    }

def find_best_threshold(y_true, y_prob) -> Tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.5
    best_f1 = -1.0

    for i, thr in enumerate(thresholds):
        p = precisions[i]
        r = recalls[i]
        if p+r == 0:
            f1 = 0.0
        else:
            f1 = 2 * (p * r) / (p + r)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold  = float(thr)

    return best_threshold, float(best_f1)

def extract_intervals(binary_sequence: np.ndarray) -> List[Tuple[int, int]]:
    intervals = []
    in_interval = False
    start = None

    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_interval:
            in_interval = True
            start = i
        elif val == 0 and in_interval:
            in_interval = False
            intervals.append((start, i))

    if in_interval:
        intervals.append((start, len(binary_sequence) - 1))

    return intervals

def compute_event_level_metrics(
        incident_sequence: np.ndarray, 
        alert_times: np.ndarray,
        decision_times: np.ndarray,
) -> Dict:
    incident_intervals = extract_intervals(incident_sequence)
    alert_set = set(alert_times.tolist()) if len(alert_times) > 0 else set()

    detected = 0
    lead_times = []

    for start, end in incident_intervals:
        pre_alerts = [t for t in alert_set if t < start]
        if len(pre_alerts)>0:
            latest_pre_alert = max(pre_alerts)
            detected += 1
            lead_times.append(start - latest_pre_alert)
    
    event_recall = detected / len(incident_intervals) if incident_intervals else 0.0

    non_incident_times = set(np.where(incident_sequence == 0)[0].tolist())
    false_positive_alerts = [t for t in alert_times if t in non_incident_times]
    false_positive_rate = len(false_positive_alerts) / max(1, len(non_incident_times))

    return {
        'incident_count': len(incident_intervals),
        'detected_incidents': detected,
        'event_recall': float(event_recall),
        'mean_lead_time': float(np.mean(lead_times)) if lead_times else 0.0,
        'median_lead_time': float(np.median(lead_times)) if lead_times else 0.0,
        'false_positive_rate': float(false_positive_rate),
    }