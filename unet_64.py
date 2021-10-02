import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size-target_size
    delta = delta//2
    return tensor[:, :, delta: tensor_size-delta, delta: tensor_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )

        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )

        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )

    def forward(self, image):
        x1_1 = self.down_conv_1(image)
        x1_2 = self.max_pool_2x2(x1_1)
        x2_1 = self.down_conv_2(x1_2)
        x2_2 = self.max_pool_2x2(x2_1)
        x3_1 = self.down_conv_3(x2_2)
        x3_2 = self.max_pool_2x2(x3_1)
        x4_1 = self.down_conv_4(x3_2)
        x4_2 = self.max_pool_2x2(x4_1)
        x5_1 = self.down_conv_5(x4_2)

        x = self.up_trans_1(x5_1)
        print(x.size())
        print(x4_1.size())
        y = crop_img(x4_1, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x3_1, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(x2_1, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(x1_1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.out(x)
        # print(x.size())
        return x

data = torch.rand((1,1,572,572))
model = UNet()
model(data)

# print(model(data))