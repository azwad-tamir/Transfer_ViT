import torch
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
from transformers import TrainerCallback
from copy import deepcopy
from transformers import DeiTFeatureExtractor, DeiTForImageClassification
import os
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
import pickle
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'

# Loading dataset:
dataset_train = load_dataset("imagefolder", data_dir="./HAM10000/train/")
dataset_test = load_dataset("imagefolder", data_dir="./HAM10000/test/")
dataset_val = load_dataset("imagefolder", data_dir="./HAM10000/val/")

# Loading feature extractor:
model_name_or_path = 'facebook/deit-base-distilled-patch16-224'
feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name_or_path)

# Data transfromation:
def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

prepared_train_ds = dataset_train.with_transform(transform)
prepared_test_ds = dataset_test.with_transform(transform)
prepared_val_ds = dataset_val.with_transform(transform)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
pred_all = []
labels_all = []
metric = load_metric("accuracy")
def compute_metrics(p):
    print("CUSTOM1: ", np.argmax(p.predictions, axis=1).shape, "CUSTOM2: ", p.label_ids.shape)
    pred_all.append(list(np.argmax(p.predictions, axis=1)))
    labels_all.append(list(p.label_ids))
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels = dataset_train['train'].features['label'].names

# Initializing model:
model = DeiTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

training_args = TrainingArguments(
  output_dir="./DeiT-pretrained",
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=20,
  fp16=True,
  save_strategy= "steps",
  save_steps=20,
  eval_steps=20,
  logging_steps=20,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train_ds["train"],
    eval_dataset=prepared_val_ds["train"],
    tokenizer=feature_extractor,
)

# Training:
trainer.add_callback(CustomCallback(trainer))
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


# Evaluation:
# metrics = trainer.evaluate(prepared_test_ds['train'])
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)
#

train_preds = []
val_preds = []
train_labels = []
val_labels = []
test_preds = []
test_labels = []
for i in range(0,len(pred_all)):
    if i%2 == 0:
        train_preds.append(deepcopy(pred_all[i]))
        train_labels.append(deepcopy(labels_all[i]))
    else:
        val_preds.append(deepcopy(pred_all[i]))
        val_labels.append(deepcopy(labels_all[i]))


# Test:
metrics = trainer.evaluate(prepared_test_ds['train'])
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

test_preds = deepcopy(pred_all)
test_labels = deepcopy(labels_all)

# Saving matrices:
metrices = [train_preds, train_labels, val_preds, val_labels, test_preds, test_labels]
with open("./final_models_pretrained/DeiT_metrices.pk", "wb") as fp:   #Pickling
    pickle.dump(metrices, fp)

# # Loading Pickle file:
# with open('./final_models_pretrained/DeiT_metrices.pk', 'rb') as f:
#     metrices = pickle.load(f)

# Plotting training and val accuracy with training steps:
train_preds = metrices[0]
train_labels = metrices[1]
val_preds = metrices[2]
val_labels = metrices[3]
train_accuracy = []
val_accuracy = []

for i in range(0,len(train_preds)):
    sum = 0
    for j in range(0,len(train_preds[i])):
        if train_preds[i][j] == train_labels[i][j]:
            sum+=1
    train_accuracy.append(sum/len(train_preds[i]))

    sum = 0
    for j in range(0, len(val_preds[i])):
        if val_preds[i][j] == val_labels[i][j]:
            sum += 1
    val_accuracy.append(sum / len(val_preds[i]))

plt.plot(range(0,len(val_accuracy)), val_accuracy, color='b', label='Validation accuracy')
plt.plot(range(0,len(train_accuracy)), train_accuracy, color='r', label='Training accuracy')
plt.title("Training and Validation accuracy")
plt.xlabel("Steps:")
plt.ylabel("Accuracy:")


# Precsion and recall:
print("\n\nDeiT MODEL:  ")
print(classification_report(labels_all[-1], pred_all[-1],digits=4))
# Confusion Matrix:
y_true = labels_all[-1]
y_pred = pred_all[-1]
data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
plt.title("Confusion matrix for DeiT model")
plt.show()