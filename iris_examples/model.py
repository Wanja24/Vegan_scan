from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, NodeConfig, TabNetModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
import pandas as pd
import numpy as np

# install: `pip install -U pytorch_tabular`
# https://github.com/manujosephv/pytorch_tabular'
# source for code: https://www.kaggle.com/code/devishu14/tabulardata-pytorch-tabular#%F0%9F%8C%B9-iris-flower-dataset


# just splits into training and test. NO validation split!
# todo: rewrite to train, val test split
def load_classification_data(df, target_col, test_size):
    torch_data = np.array(df.drop(target_col, axis=1))
    torch_labels = np.array(df[target_col])
    data = np.hstack([torch_data, torch_labels.reshape(-1, 1)])
    gen_names = [f"feature_{i}" for i in range(data.shape[-1])]
    col_names = gen_names
    col_names[-1] = "target"
    data = pd.DataFrame(data, columns=col_names)
    cat_col_names = [x for x in gen_names[:-1] if len(data[x].unique()) < 10]
    num_col_names = [x for x in gen_names[:-1] if x not in [target_col] + cat_col_names]
    test_idx = data.sample(int(test_size * len(data)), random_state=42).index
    test = data[data.index.isin(test_idx)]
    train = data[~data.index.isin(test_idx)]
    return (train, test, ["target"], cat_col_names, num_col_names)


# reading data
data2 = pd.read_csv('../data/iris.csv')
# encoding labels
data2['variety'] = data2['variety'].astype('category').cat.codes

train, test, target_col, cat_col_names, num_col_names = load_classification_data(data2, 'variety', 0.2)

data_config = DataConfig(
    # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is
    # not implemented
    target=['target'],
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform="quantile_normal",
    normalize_continuous_features=True
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=32,
    max_epochs=10,
)
optimizer_config = OptimizerConfig()
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="4096-4096-512",  # Number of nodes in each layer
    activation="LeakyReLU",  # Activation between each layers
    learning_rate=1e-3,
    metrics=["accuracy"]
)
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tabular_model.fit(train=train, test=test)

"""
Epoch 9/9  ------------------ 4/4 0:00:00 • 0:00:00 9.15it/s loss: 0.517       
                                                             train_loss: 0.394 
                                                             valid_loss: 0.421 
                                                             valid_accuracy:   
                                                             0.792             
                                                             train_accuracy:   
                                                             0.802   
"""