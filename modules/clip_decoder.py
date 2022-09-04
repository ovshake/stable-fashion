import torch
import clip
import PIL
from torch import nn
from PIL import Image
from torch import optim




class CLIPDecoder:
    def __init__(self, clip_model, epochs=600, logger_patience=100, device="cuda"):
        self.model = clip_model
        self.epochs = epochs
        self.logger_patience = logger_patience
        self.device = device
        for m in self.model.parameters():
            m.requires_grad = False

    def get_optimizer(self, learnable_features):
        learnable_features.requires_grad = True
        optimizer = optim.SGD([learnable_features], lr=1e-3, momentum=0.99)
        return optimizer

    def get_embedding(self, image, text):
        image_features = self.model.encode_image(image)
        tokenized_text = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(tokenized_text).squeeze()
        text_features = text_features.repeat((image.shape[0],) + (1, ) * len(text_features.shape)) # repeating with number of batch size.
        optimizer = self.get_optimizer(text_features)
        cossim = nn.CosineSimilarity()
        for epoch in range(self.epochs):
            similarity = cossim(image_features, text_features)
            loss =  (1 - similarity).pow(2).mean()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % self.logger_patience == 0:
                print(f"Epoch: {epoch} | Loss: {loss.item():.3f} | Cos Sim: {similarity.mean().item():.2f}")

        return text_features.detach().cpu().numpy()



if __name__ == '__main__':
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14", device=device)
    image_path = "/data/abhishek/projects/ACGPN/Data_preprocessing/test_color/000048_1.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    text = clip.tokenize(["A picture of cloth"]).to(device)
    print(text)
    text_features = model.encode_text(text)
    text_features.requires_grad = True
    optimizer = optim.SGD([text_features], lr=1e-4, momentum=0.99)
    cossim = nn.CosineSimilarity()
    criterion = nn.MSELoss()
    for epoch in range(600):
        loss =  (1 - cossim(image_features, text_features)).pow(2).mean()
        loss.backward()
        optimizer.step()









