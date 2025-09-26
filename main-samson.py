'''
@Project ：MG-Net 
@File    ：main-samson.py
@IDE     ：PyCharm 
@Author  ：王一梵
@Date    ：2025/9/26 10:42 
'''
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import scipy.io as sio
import torchvision.transforms as transforms

from utils.Wavelet_Transform_Separation import wavelet_transform_separation
from kan import *

from utils.attention import MultiHeadAttention
from utils.tools import *
from utils.Dir_CVAE import *

device = torch.device("cuda:0")

torch_seed = 114514
numpy_seed = 8673862  #
torch.manual_seed(torch_seed)
torch.cuda.manual_seed(torch_seed)
np.random.seed(numpy_seed)

# Load DATA
data = sio.loadmat("samson_dataset.mat")
abundance_GT = torch.from_numpy(data["A"])  # true abundance    3*9025
original_HSI = torch.from_numpy(data["Y"])  # mixed abundance    156*9025

# VCA_endmember and GT
VCA_endmember = data["M1"]  # 156*3
GT_endmember = data["M"]  # 156*3

band_Number = original_HSI.shape[0]  # 156
endmember_number, pixel_number = abundance_GT.shape  # 3    9025
col = 95
original_HSI = torch.reshape(original_HSI, (band_Number, col, col))  # 156*95*95
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))  # 3*95*95

batch_size = 1
EPOCH = 400
drop_out = 0.1
learning_rate = 0.002
print(
    "batchsize: %d | epoch: %d  | drop_out: %.3f | learning_rate: %.3f" % (batch_size, EPOCH, drop_out, learning_rate),
    '\r')
start_time = time.time()


# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(abundance_input.squeeze(0), (endmember_number, col, col))
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input):
    plt.figure(figsize=(20, 10))
    for i in range(0, endmember_number):
        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_input[i, :, :], cmap="jet")

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_GT_input[i, :, :], cmap="jet")
    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT):
    plt.figure(figsize=(20, 5))
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], color="b")
        plt.plot(endmember_GT[:, i], color="r")
    plt.show()


# change the index of abundance and endmember true
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[RMSE_index, :, :]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]
    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, transform=None):
        self.img = img.float()
        # self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        # return self.img, self.gt
        return self.img

    def __len__(self):
        return 1


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


def conv11(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


# my net
class MG_Net_samson(nn.Module):
    def __init__(self):
        super(MG_Net_samson, self).__init__()
        self.linear_encoder_layer = nn.Sequential(
            conv33(band_Number, 96),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
            nn.Softmax(dim=2)
        )
        self.linear_decoder_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )
        self.linear_decoder_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )
        self.nonlinear_encoder_layer = nn.Sequential(
            conv11(band_Number, 96),
            nn.BatchNorm2d(96),
            nn.Tanh(),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.BatchNorm2d(48),
            nn.Tanh(),
            conv11(48, band_Number),
            nn.BatchNorm2d(band_Number),
            nn.Tanh(),
        )
        self.maxpool = nn.Sequential(
            nn.Conv2d(in_channels=band_Number, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
        )
        self.KANblock = nn.Sequential(
            KAN(width=[16, 5, 32], grid=3, k=5, device=device)
        )
        self.upsample2 = nn.Upsample([col, col], mode='bilinear', align_corners=False)
        self.vtrans = MultiHeadAttention(band_Number, band_Number, band_Number, 300, 30, drop_out)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=band_Number, out_channels=300, kernel_size=1, stride=1),
            nn.BatchNorm2d(300),
            nn.ReLU(),
        )
        self.Conv = nn.Sequential(
            conv33(32, 128),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            conv33(128, band_Number),
            nn.BatchNorm2d(band_Number),
            nn.Tanh(),
        )
        self.EDLClassifier = EDLClassifier(self.encoder, 300, 100, endmember_number, drop_out)

    def forward(self, x, y):
        x1 = self.linear_encoder_layer(x)
        linear_part = self.linear_decoder_layer1(x1)
        evidence = self.EDLClassifier(x)
        probabilities, total_uncertainty, beliefs = self.EDLClassifier.predict(evidence)
        linear_part1 = self.linear_decoder_layer2(probabilities)
        y_T = y.permute(0, 2, 3, 1)
        weight = self.vtrans(y_T, y_T, y_T)
        y = self.maxpool(y)
        y = torch.reshape(y, [16, -1]).T
        y = self.KANblock(y)
        y = torch.reshape(y.T, [1, 32, col // 4, col // 4])
        y = self.upsample2(y)
        nonlinear_part = self.Conv(y)
        reconstruction = linear_part + nonlinear_part * weight
        return reconstruction, x1, evidence, probabilities, total_uncertainty, linear_part1 


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)
    return abundance_loss


MSE = torch.nn.MSELoss(reduction='mean')
KLD_loss = UnsupervisedKLDivergenceLoss(num_class=endmember_number)
criterion = nn.CrossEntropyLoss()


# weights_init
def weights_init(m):
    nn.init.kaiming_normal_(net.linear_encoder_layer[0].weight.data)
    nn.init.kaiming_normal_(net.linear_encoder_layer[4].weight.data)
    nn.init.kaiming_normal_(net.linear_encoder_layer[8].weight.data)


# load data
train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False
)

net = MG_Net_samson().cuda()
# edgeLoss = EdgeLoss()
# weight init
# net.apply(weights_init)

# decoder weight init by VCA
model_dict = net.state_dict()
model_dict["linear_decoder_layer1.0.weight"] = endmember_init
model_dict["linear_decoder_layer2.0.weight"] = endmember_init

net.load_state_dict(model_dict)  # 预训练的参数权重加载到新的模型之中

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)

Y_samson = original_HSI.detach().cpu().squeeze().numpy()
Y_samson = Y_samson.transpose(1, 2, 0)
linear_part, nonlinear_part = wavelet_transform_separation(Y_samson, 0.01)
linear_part = linear_part.transpose(2, 0, 1)
nonlinear_part = nonlinear_part.transpose(2, 0, 1)
linear_part = torch.from_numpy(linear_part).unsqueeze(0).type(torch.cuda.FloatTensor)
nonlinear_part = torch.from_numpy(nonlinear_part).unsqueeze(0).type(torch.cuda.FloatTensor)

# train
for epoch in range(EPOCH):
    for i, x in enumerate(train_loader):
        # scheduler.step()
        x = x.cuda()
        reconstruction_result3, en_abundance3, evidence, probabilities, total_uncertainty, linear_recon = net(
            linear_part,
            nonlinear_part)  #
        E = net.state_dict()["linear_decoder_layer1.0.weight"]
        abundanceLoss3 = reconstruction_SADloss(x, reconstruction_result3)
        MSELoss3 = MSE(x, reconstruction_result3)
        abundanceLoss3_linear = SADloss_weigh(linear_recon, linear_part, 1 - total_uncertainty)
        MSELoss3_linear = rmse_loss(linear_recon, linear_part, 1 - total_uncertainty)
        KLD_Loss = 0.1 * KLD_loss(evidence)
        loss_orth = 0.005 * orthogonal_loss(x1.squeeze(), E.squeeze().T)  # 0.001
        loss_CE = 10 * criterion(probabilities, x1)  # 1

        Overall_recon_loss = 5 * abundanceLoss3 + 0.01 * MSELoss3
        Linear_recon_loss = (5 * abundanceLoss3_linear + 0.01 * MSELoss3_linear)
        Abundance_model_loss = 1 * (Linear_recon_loss + KLD_Loss + loss_CE)
        total_loss = Overall_recon_loss + Abundance_model_loss + loss_orth
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print("******MSELoss3: %.4f" % MSELoss3.item(), "******abundanceLoss3: %.4f" % abundanceLoss3.item())
            print("Epoch:", epoch + 1, "| loss: %.4f" % total_loss.cpu().data.numpy(),
                  "| Overall_recon_loss: %.4f" % Overall_recon_loss.item(),
                  "| Linear_recon_loss: %.4f" % Linear_recon_loss.item(),
                  "| KLD_Loss: %.4f" % KLD_Loss.item(),
                  "| loss_CE: %.4f" % loss_CE.item(),
                  "| loss_orth1: %.4f" % loss_orth1.item(),
                  "| loss_orth2: %.4f" % loss_orth2.item(), )

end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time:.4f} 秒")

# 保存模型权重
torch.save(net.state_dict(), "./results/samson/model_weights.pth")

# 加载模型权重
# net.load_state_dict(torch.load("./results/samson/model_weights.pth"))
# net.eval()

# 评估
reconstruction_result3, en_abundance3, _, _, _, _ = net(linear_part, nonlinear_part)
decoder_para = net.state_dict()["linear_decoder_layer1.0.weight"].cpu().numpy()
# print("endmember",decoder_para.shape)
decoder_para = np.mean(np.mean(decoder_para, -1), -1)
en_abundance, abundance_GT = norm_abundance_GT(en_abundance3, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

en_abundance, decoder_para, RMSE_abundance, SAD_endmember = arange_A_E(
    en_abundance, abundance_GT, decoder_para, GT_endmember
)

print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())
sio.savemat('results/samson/samson.mat', {'A': en_abundance, 'E': decoder_para})
plot_abundance(en_abundance, abundance_GT)
plot_endmember(decoder_para, GT_endmember)
