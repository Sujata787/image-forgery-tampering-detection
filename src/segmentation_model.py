import torch
import torch.nn as nn


# ==========================================
# DOUBLE CONVOLUTION BLOCK
# ==========================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ==========================================
# LIGHTWEIGHT U-NET
# ==========================================

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):

        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 16)

        self.encoder2 = DoubleConv(16, 32)

        self.encoder3 = DoubleConv(32, 64)

        self.encoder4 = DoubleConv(64, 128)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )

        self.decoder4 = DoubleConv(256, 128)

        self.upconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.decoder3 = DoubleConv(128, 64)

        self.upconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.decoder2 = DoubleConv(64, 32)

        self.upconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2
        )

        self.decoder1 = DoubleConv(32, 16)

        self.output = nn.Conv2d(
            16,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)

        e2 = self.encoder2(self.pool(e1))

        e3 = self.encoder3(self.pool(e2))

        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.upconv4(b)

        d4 = torch.cat([d4, e4], dim=1)

        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)

        d3 = torch.cat([d3, e3], dim=1)

        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)

        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.decoder1(d1)

        return self.output(d1)


# ==========================================
# MODEL BUILDER
# ==========================================

def build_segmentation_model():

    return UNet(
        in_channels=3,
        out_channels=1
    )


# ==========================================
# MODEL TEST
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("LIGHTWEIGHT U-NET MODEL TEST")
    print("=" * 55)

    model = build_segmentation_model()

    dummy_input = torch.randn(1, 3, 128, 128)

    with torch.no_grad():

        output = model(dummy_input)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nInput shape :", dummy_input.shape)

    print("Output shape:", output.shape)

    print(f"Parameters  : {total_parameters:,}")

    print("\nLIGHTWEIGHT U-NET TEST COMPLETED.")