# Predictive Alerting for Cloud Metrics
## For JETBRAINS Internship: Predictive alerting for cloud metrics

This repository contains a prototype predictive alerting system for time-series metrics.
The goal is to predict whether an incident will occur within the next `H` time steps based on the previous `W` observations of one or more metrics.

The implementation uses a sliding-window formulation and trains an LSTM-based binary classifier on synthetic multivariate time-series data with labeled incident intervals.

## Task

The implemented solution addresses the following problem:
> Predict whether an incident will occur within the next `H` time steps based on the previous `W` time steps od one or more metrics, using a sliding-window formulation and a standard machine-learning framework.

The focus of this repository is on:
- correct problem formulation,
- model selection,
- training and evaluation,
- threshold-based alert generation,
- analysis of strengths and limitations.

## Problem formulation

Given:
- a multivariate metric sequence `x[t]`
- a lookback window of size `W`
- a prediction horizon `H`
training samples are constructed as:
- **input**: the previous `W` observations, `x[t-W:t]`
- **label**: `1` if an incident occurs at least once in the interval `[t, t+H)`, otherwise `0`

This formulation matches the predictive alerting use case:
the model should raise an alert *before* an incident starts, rather than merely detect it after it has already begun.

## Dataset

For this prototype, I used a **synthetic multivariate time series** with labeled incident intervals.

The synthetic dataset was designed to mimic several characteristics of cloud-service metrics:
- multiple correlated signals,
- noisy behaviour,
- regime changes,
- localized pre-incident ramps,
- incident intervals with abnormal metric behavior.

Synthetic data was chosen to keep the repository self-contained and reproducible while still supporting controlled experiments with event-level evaluation.

## Modeling choices

### Why binary  incident prediction?
Instead of forecasting raw future metric values and then deriving alerts from forecast errors, the model directly predicts whether an incident will occur within the next `H` steps. It keeps objective aligned with the operational goal of alerting.

### Why LSTM?
The core model I used in the project, is an **LSTM-based classifier** implemented in PyTorch.
Reasons for choosing it:
- the task is sequential,
- recent temporal context matters,
- it is simple enough for compact prototype while still capturing temporal dependencies.

### Input/output setup
- Input: a window of lenght `W=60`
- Output: probability that an incident will occur within the next `H=6` steps

## Training setup
### Data split
The dataset is split **chronologically** into:
- training set: 70%
- validation set: 15%
- test set: 15%
A chronological split is used to avoid temporal leakage and reflect deployment on unseen data.

### Preprocessing
Feature scalling is fitted only on the training windows and then applied to validation and test windows.

### Loss
The model is trained using `BCEWithLogitsLoss`

### Hyperparameters
Final configuration used for the reported results:

- `window_size = 60`
- `horizon = 6`
- `hidden_size = 64`
- `num_layers = 1`
- `dropout = 0.1`
- `batch_size = 64`
- `epochs = 12`
- `learning_rate = 0.001`
- `weight_decay = 0.001`
- `positive_class_weight = 1.0`

## Evaluation setup
The model outputs an incident probability for each sliding window.

### Threshold selection
A decision threshold is selected on the **validation set** by maximizing F1 score and then applied unchanged to the test set.

### Reported metrics
Pointwise classification metrics, computed per sliding window:
- precision
- recall
- F1 score
- PR AUC

Event-level aligned metrics, more aligned with real alerting use cases:
- event recall (fraction of incident intervals for which at least one alert was raised before the incident start)
- mean lead time (average number of time steps between first alert and incident start)
- median lead time
- false positive rate

## Results
Final test-set results:
- `"threshold": 0.9657440185546875`,
- `"precision": 0.3669724770642202`,
- `"recall": 0.5031446540880503`,
- `"f1_score": 0.4244031830238727`,
- `"pr_auc": 0.4689566177706477`,
- `"event_recall": 1.0`,
- `"mean_lead_time": 36.75`,
- `"median_lead_time": 30.0`,
- `"false_positive_rate": 0.08527131782945736`

A visualization of the predicted probabilities on the test set is included in `results/predictions.png`. 

### Interpretation
The final model detected all incident intervals on the test split at the event level, while keeping the false-positive rate relatively low and providing substantial lead time before incident onset. The plot included in `results/predictions.png` illustrates the qualitative behavior behind the reported metrics by showing predicted probabilities, ground-truth window labels, and the selected alert threshold on the test split.

## Installation and running the project
Installing the dependencies:
```bash
pip install -r requirements.txt
```
Training the model:
```bash
python -m src.train
```
Evaluating the model:
```bash
python -m src.evaluate
```
