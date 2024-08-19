import torch
import torch.optim as optim
import argparse
from dataset import get_dataloaders
from ae import AutoEncoder
from unet import UNet
from ukan import UKan
from loss import AutoencoderLoss, SegmentationLoss
import matplotlib.pylab as plt
import logging
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    return logging.getLogger()


def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--model', type=str, default='UNet', choices=['UKan', 'UNet'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def save_combined_images(images, autoencoder_A, autoencoder_B, output_dir='/root/autodl-tmp/figure'):
    # 创建输出目录
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将图像扁平化为一维以喂入自动编码器
    images_ae = images.view(images.size(0), -1)

    with torch.inference_mode():
        recon_A = autoencoder_A(images_ae)
        # recon_B = autoencoder_B(images_ae)

    # 将输出重新调整为原始图像的形状
    output_a = recon_A.view(-1, images.size(1), images.size(2))
    # output_b = recon_B.view(-1, images.size(1), images.size(2))

    # 绘制并保存图像
    for i in range(images.size(0)):
        plt.figure(figsize=(15, 5))  # 设置画布大小
        plt.subplot(131)
        plt.imshow(images[i].cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')  # 关闭坐标轴

        plt.subplot(132)
        plt.imshow(output_a[i].cpu().numpy(), cmap='gray')
        plt.title('Reconstructed Image A')
        plt.axis('off')

        # plt.subplot(133)
        # plt.imshow(output_b[i].cpu().numpy(), cmap='gray')
        # plt.title('Reconstructed Image B')
        # plt.axis('off')

        plt.tight_layout()  # 调整子图布局
        plt.savefig(f'{output_dir}/combined_image_{i + 1}.png')  # 保存图像
        plt.close()  # 关闭画布以节省内存

def create_target_masks(images, autoencoder_A, autoencoder_B):
    images_ae = images.view(images.size(0), -1)
    # 使用预训练的AutoEncoder对图像进行重构，得到的重构图像作为伪标签
    with torch.inference_mode():
        recon_A = autoencoder_A(images_ae)
        recon_B = autoencoder_B(images_ae)

    # 由于重构图像是连续的，我们需要将其二值化来近似目标掩模
    # 二值化的阈值可以根据实际情况调整，这里简单使用0.5作为例子
    threshold = 0.5
    mask_A = (recon_A > threshold).float()
    mask_B = (recon_B > threshold).float()

    # 将两个掩模合并为多类分割的目标
    # 注意：这里假设mask_A和mask_B没有重叠的部分，实际情况可能需要调整处理重叠逻辑
    target_masks = mask_A + mask_B * 2  # 为两个结构赋予不同的标签

    return target_masks


def main():
    args = parse_args()

    # 设置日志
    log_dir = '/root/autodl-tmp/logs'
    logger = setup_logger(log_dir)

    logger.info(f"Starting training with arguments: {args}")

    # 超参数
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    input_channel = 1
    num_classes = 3
    epoch_ae = 900

    # 创建数据加载器
    dataloaderA, dataloaderB, dataloaderAB = get_dataloaders()
    logger.info("Data loaders created")
    # mean_a, std_a = compute_mean_std(dataloaderA)
    # mean_b, std_b = compute_mean_std(dataloaderB)
    # mean_ab, std_ab = compute_mean_std(dataloaderAB)


    # 初始化模型
    autoencoder_A = AutoEncoder().to(device)
    autoencoder_B = AutoEncoder().to(device)
    if args.model == 'UNet':
        segmentation_model = UNet(n_channels=input_channel, n_classes=num_classes).to(device)
    elif args.model == 'UKan':
        segmentation_model = UKan(n_channels=input_channel, n_classes=num_classes).to(device)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")

    logger.info(f"Models initialized: AutoEncoder A, AutoEncoder B, and {args.model}")

    # 定义优化器
    optimizer_A = optim.Adam(autoencoder_A.parameters(), lr=args.lr)
    optimizer_B = optim.Adam(autoencoder_B.parameters(), lr=args.lr)
    optimizer_seg = optim.Adam(segmentation_model.parameters(), lr=args.lr)
    logger.info("Optimizers created")

    # 定义学习率调度器
    scheduler_A = ReduceLROnPlateau(optimizer_A, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler_B = ReduceLROnPlateau(optimizer_B, mode='min', factor=0.1, patience=10, verbose=True)
    logger.info("Learning rate schedulers created")

    # 定义损失函数
    autoencoder_loss = AutoencoderLoss()
    segmentation_loss = SegmentationLoss()
    logger.info("Loss functions defined")

    # loss_values_a = []
    # loss_values_b = []
    # lowest_loss_a = float('inf')
    # lowest_loss_b = float('inf')
    # # 训练循环
    # for epoch in range(epoch_ae):
    #     total_loss_A = 0.0
    #     total_loss_B = 0.0
    #     for items in dataloaderA:
    #         images = items['image']
    #         images = images.to(device)
    #         images = images.float()
    #
    #         # 训练 AutoEncoder A
    #         optimizer_A.zero_grad()
    #         images = images.view(images.size(0), -1)
    #         recon_A = autoencoder_A(images)
    #         loss_A = autoencoder_loss(recon_A, images)
    #         loss_A.backward()
    #         optimizer_A.step()
    #         total_loss_A += loss_A.item()
    #
    #     for items in dataloaderB:
    #         images = items['image']
    #         images = images.to(device)
    #         images = images.float()
    #
    #         # 训练 AutoEncoder B
    #         optimizer_B.zero_grad()
    #         images = images.view(images.size(0), -1)
    #         recon_B = autoencoder_B(images)
    #         loss_B = autoencoder_loss(recon_B, images)
    #         loss_B.backward()
    #         optimizer_B.step()
    #         total_loss_B += loss_B.item()
    #
    #     avg_loss_A = total_loss_A / len(dataloaderA)
    #     avg_loss_B = total_loss_B / len(dataloaderB)
    #
    #     # 更新学习率
    #     scheduler_A.step(avg_loss_A)
    #     scheduler_B.step(avg_loss_B)
    #
    #     # 保存平均损失
    #     loss_values_a.append(avg_loss_A)
    #     loss_values_b.append(avg_loss_B)
    #
    #     # print(f"Current learning rate A: {optimizer_A.param_groups[0]['lr']:.6f}")
    #     # print(f"Current learning rate B: {optimizer_B.param_groups[0]['lr']:.6f}")
    #     # print(f"Epoch [{epoch + 1}/{epoch_ae}], Loss A: {avg_loss_A:.4f}, Loss B: {avg_loss_B:.4f}")
    #     logger.info(f"Current learning rate A: {optimizer_A.param_groups[0]['lr']:.6f}")
    #     logger.info(f"Current learning rate B: {optimizer_B.param_groups[0]['lr']:.6f}")
    #     logger.info(f"Epoch [{epoch + 1}/{epoch_ae}], Loss A: {avg_loss_A:.4f}, Loss B: {avg_loss_B:.4f}")
    #     if loss_A < lowest_loss_a:
    #         lowest_loss_a = loss_A
    #         torch.save(autoencoder_A.state_dict(), f'/root/autodl-tmp/result/unet/models/autoencoder_A.pth')
    #         # print(f'A: New lowest average loss {lowest_loss_a:.4f} at epoch {epoch + 1}, model saved.')
    #         logger.info(f'A: New lowest average loss {lowest_loss_a:.4f} at epoch {epoch + 1}, model saved.')
    #
    #     if loss_B < lowest_loss_b:
    #         lowest_loss_b = loss_B
    #         torch.save(autoencoder_B.state_dict(), f'/root/autodl-tmp/result/unet/models/autoencoder_B.pth')
    #         # print(f'B: New lowest average loss {lowest_loss_b:.4f} at epoch {epoch + 1}, model saved.')
    #         logger.info(f'B: New lowest average loss {lowest_loss_b:.4f} at epoch {epoch + 1}, model saved.')
    #
    # plt.plot(loss_values_a, label='Loss A')
    # plt.plot(loss_values_b, label='Loss B')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.legend()
    # plt.savefig('loss_curve.png')  # 保存图像

    autoencoder_A.load_state_dict(torch.load('/root/autodl-tmp/result/unet/models/autoencoder_A.pth'))
    autoencoder_B.load_state_dict(torch.load('/root/autodl-tmp/result/unet/models/autoencoder_B.pth'))
    autoencoder_A.eval()
    autoencoder_B.eval()
    logger.info("Loaded best models for AutoEncoder A and B")

    logger.info("AutoEncoder Training completed")

    for items in dataloaderA:
        images = items['image']
        images = images.to(device)
        images = images.float()
        save_combined_images(images, autoencoder_A, autoencoder_B)

    for epoch in range(args.epochs):
        for items in dataloaderAB:
            images = items['image']
            images = images.to(device)
            images = images.float()
            optimizer_seg.zero_grad()

            images_ae = images.view(images.size(0), -1)
            with torch.no_grad():
                latent_A = autoencoder_A.encoder(images_ae)
                latent_B = autoencoder_B.encoder(images_ae)

            # 使用预训练的AutoEncoder的输出作为伪目标掩模
            target = create_target_masks(images, autoencoder_A, autoencoder_B).to(device)
            # 将两个潜在特征concatenate后输入到分割模型
            concatenated_features = torch.cat((latent_A, latent_B), dim=1)  # 假设编码器的输出可以直接拼接

            # 训练分割模型
            seg_outputs = segmentation_model(concatenated_features)
            loss_seg = segmentation_loss(seg_outputs, target)
            loss_seg.backward()
            optimizer_seg.step()

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss Seg: {loss_seg.item():.4f}")

    # 保存模型
    torch.save(segmentation_model.state_dict(), f'{args.model}.pth')


if __name__ == "__main__":
    main()
