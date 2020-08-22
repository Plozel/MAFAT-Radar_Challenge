from src.Classifier.RadarClassifier import RadarClassifier
from pytorch_lightning import Trainer

if __name__ == "__main__":
    model = RadarClassifier()
    trainer = Trainer(gpus=1, num_nodes=1, max_epochs=50)
    trainer.fit(model)
