# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch

from utils.data.spatiotemporal_csv_data import SpatioTemporalCSVDataModule


SupervisedDataModule = SpatioTemporalCSVDataModule


__all__ = [
    "SupervisedDataModule",
    "SpatioTemporalCSVDataModule",
]
