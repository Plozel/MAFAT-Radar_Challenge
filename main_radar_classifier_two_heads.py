from src.Classifier.RadarClassifier import RadarClassifierTwoHeads
from pytorch_lightning import Trainer

if __name__ == "__main__":
    checkpoint_path = 'lightning_logs/version_0/checkpoints/epoch=0.ckpt'
    model = RadarClassifierTwoHeads()
    model.convert_to_two_heads(checkpoint_path)
    trainer = Trainer(gpus=1, num_nodes=1, max_epochs=5)
    trainer.fit(model)