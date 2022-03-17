import torch.nn as nn 
import timm 

class ResNetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = timm.create_model(
            cfg['MODEL']['MODEL_NAME'], 
            pretrained=cfg['MODEL']['PRETRAINED']
            )
        
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128, cfg['MODEL']['NUM_CLASSES'])

    def forward(self, image):
        x = self.model(image)
        x = self.dropout(x)
        output = self.out(x)
        return output

class ResNextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = timm.create_model(
            cfg['MODEL']['MODEL_NAME'], 
            pretrained=cfg['MODEL']['PRETRAINED']
            )
        
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, cfg['MODEL']['NUM_CLASSES'])

    def forward(self, image):
        x = self.model(image)
        x = self.dropout(x)
        output = self.out(x)
        return output

class VGGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = timm.create_model(
            cfg['MODEL']['MODEL_NAME'], 
            pretrained=cfg['MODEL']['PRETRAINED']
            )
        
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128, cfg['MODEL']['NUM_CLASSES'])

    def forward(self, image):
        x = self.model(image)
        x = self.dropout(x)
        output = self.out(x)
        return output

class ViTModel(nn.Module):
    def __init__(self, cfg):

        super(ViTModel, self).__init__()

        self.model = timm.create_model(
            cfg['MODEL']['MODEL_NAME'], 
            pretrained=cfg['MODEL']['PRETRAINED']
            )

        self.model.head = nn.Linear(
            self.model.head.in_features, 
            cfg['MODEL']['NUM_CLASSES']
        )

    def forward(self, x):
        x = self.model(x)
        return x

class EffNetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = timm.create_model(
            cfg['MODEL']['MODEL_NAME'], 
            pretrained=cfg['MODEL']['PRETRAINED']
            )
        
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, cfg['MODEL']['NUM_CLASSES'])

    def forward(self, image):
        x = self.model(image)
        x = self.dropout(x)
        output = self.out(x)
        return output