import torch
import numpy as np
import pandas as pd
from src.Dataset_handlers.MAFATDataset import MAFATDataset
from src.Classifier.RadarClassifier import RadarClassifier

if __name__ == "__main__":
    model = RadarClassifier()
    checkpoint_path = 'lightning_logs/version_1/checkpoints/epoch=39.ckpt'
    model.load_from_checkpoint(checkpoint_path)
    test_set = MAFATDataset(['test'])
    predictions = []
    model.eval()
    with torch.no_grad():
        for (iq_data, _) in test_set:
            y_hat = model(iq_data)
            predictions.extend(torch.exp(y_hat[:, 1]).tolist())
    submission = pd.DataFrame()
    submission['segment_id'] = test_set.target_segment_ids
    submission['prediction'] = np.array(predictions)
    # Save submission
    submission.to_csv('submission.csv', index=False)
