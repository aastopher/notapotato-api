import torch

num_channels = 3 # number of channels in the training images. For color images this is 3 (RGB)
image_size = 64 # also the size of feature maps for the generator and discriminator
z_size = 100 # size of z latent vector (i.e. size of generator input)

class Generator(torch.nn.Module):
    def __init__(self, num_gpu):
        super(Generator, self).__init__()

        self.num_gpu = num_gpu
        self.trainImgs = []
        self.losses = []
        
        tconv1 = image_size * 8 # 64 x 8 = 512 channels
        tconv2 = image_size * 4 # 64 x 4 = 256 channels
        tconv3 = image_size * 2 # 64 x 2 = 128 channels
        
        self.main = torch.nn.Sequential(
            # input Z = 100 x 1 
            torch.nn.ConvTranspose2d(in_channels = z_size, # 100
                               out_channels = tconv1, # 64 x 8 = 512 channels
                               kernel_size = 4, 
                               stride = 1, 
                               padding = 0, 
                               bias=False),
            torch.nn.BatchNorm2d(tconv1),
            torch.nn.ReLU(True),
            # feature_size = 64 / 2^4 = 64 / 16 = 4 ==> 4 x 4
            # feature_channels = 512 x 2 = 1024
            # transpose_state = 4 x 4 (1024 channels)

            torch.nn.ConvTranspose2d(in_channels = tconv1, # 64 x 8 = 512 channels
                               out_channels = tconv2, # 64 x 4 = 256 channels
                               kernel_size = 4, 
                               stride = 2, 
                               padding = 1,
                               bias=False),
            torch.nn.BatchNorm2d(tconv2),
            torch.nn.ReLU(True),
            # feature_size = 64 / 2^3 = 64 / 8 = 8 ==> 8 x 8
            # feature_channels = 256 x 2 = 512
            # transpose_state = 8 x 8 (512 channels)

            torch.nn.ConvTranspose2d(in_channels = tconv2, # 64 x 4 = 256 channels
                               out_channels = tconv3, # 64 x 2 = 128 channels
                               kernel_size = 4, 
                               stride = 2, 
                               padding = 1,
                               bias=False),
            torch.nn.BatchNorm2d(tconv3),
            torch.nn.ReLU(True),
            # feature_size = 64 / 2^2 = 64 / 4 = 16 ==> 16 x 16
            # feature_channels = 128 x 2 = 256
            # transpose_state = 16 x 16 (256 channels)

            torch.nn.ConvTranspose2d(in_channels = tconv3, # 64 x 2 = 128 channels
                               out_channels = image_size, # 64 x 64 = 3 channels
                               kernel_size = 4, 
                               stride = 2, 
                               padding = 1,
                               bias=False),
            torch.nn.BatchNorm2d(image_size),
            torch.nn.ReLU(True),
            # feature_size = 64 / 2^1 = 64 / 2 = 32 ==> 32 x 32
            # feature_channels = 64 x 2 = 128
            # transpose_state = 32 x 32 (128 channels)

            torch.nn.ConvTranspose2d(in_channels = image_size,
                               out_channels = num_channels, 
                               kernel_size = 4, 
                               stride = 2, 
                               padding = 1, 
                               bias=False),
            torch.nn.Tanh()
            # output = 64 x 64 (3 channels)
        )

    def forward(self, input):
        return self.main(input)