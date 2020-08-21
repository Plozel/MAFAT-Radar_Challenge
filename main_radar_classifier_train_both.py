from src.Classifier.RadarClassifier import RadarClassifier, RadarClassifierTwoHeads
from pytorch_lightning import Trainer

if __name__ == "__main__":
    model = RadarClassifier()
    trainer = Trainer(gpus=1, num_nodes=1, max_epochs=2)
    trainer.fit(model)

    checkpoint_path = 'last.ckpt'
    trainer.save_checkpoint(checkpoint_path)

    model = RadarClassifierTwoHeads()
    model.convert_to_two_heads(checkpoint_path)
    trainer = Trainer(gpus=1, num_nodes=1, max_epochs=1)
    trainer.fit(model)
    