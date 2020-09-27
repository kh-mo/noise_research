import os
import sys
sys.path.append(os.path.join(os.getcwd(), "anomalydetection"))
sys.path.append(os.path.join(os.getcwd(), "datasets"))

from anomalydetection.autoencoder import AutoEncoder
from datasets.mnist import get_dataset

import torch

if __name__ == "__main__":
    # config
    batch_size = 8

    # data
    train_data, test_data = get_dataset("data")
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size
    )

    next(iter(train_loader))

    # model
    model = AutoEncoder()

    # training
    for train in train_loader:
        model(train)

    save_model(model)