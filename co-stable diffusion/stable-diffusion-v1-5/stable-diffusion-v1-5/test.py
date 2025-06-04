import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题


class TestConfig:
    # Paths
    pretrained_model_path = "D:\\github\\deep-learning-from-scratch\\co-stable diffusion\\stable-diffusion-v1-5\\stable-diffusion-v1-5"
    fine_tuned_model_path = "fine_tuned_model_cpu\\final_model\\unet"
    image_dir = os.path.join(pretrained_model_path, "images", "images")
    csv_path = os.path.join(pretrained_model_path, "POEM_IMAGE.csv")

    # Test parameters
    batch_size = 1
    resolution = 512
    start_idx = 102  # Start from image 100
    end_idx = 222  # End at image 120
    seed = 42  # For reproducibility


class PoemImageTestDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, start_idx=100, end_idx=120):
        full_df = pd.read_csv(csv_path, sep='\t', encoding='utf-8-sig')

        # Select specific range of images (102-120)
        self.df = full_df.iloc[start_idx:end_idx]

        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((TestConfig.resolution, TestConfig.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_id']).strip()
        if img_name.startswith('._'):
            img_name = img_name[2:]

        base_name = os.path.splitext(img_name)[0]
        possible_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        img_path = None

        for ext in possible_extensions:
            test_path = os.path.join(self.image_dir, base_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path is None:
            image = Image.new('RGB', (512, 512), color=(0, 0, 0))
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new('RGB', (512, 512), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        poem = row['poem'] + "Chinese ink painting, traditional style"
        return {"pixel_values": image, "input_ids": poem, "image_id": row['image_id']}


def prepare_test_dataloader(config):
    dataset = PoemImageTestDataset(
        csv_path=config.csv_path,
        image_dir=config.image_dir,
        transform=transforms.Compose([
            transforms.Resize((config.resolution, config.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        start_idx=config.start_idx,
        end_idx=config.end_idx
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)


def evaluate_model(config, model_type="fine_tuned"):
    device = torch.device("cpu")

    # Load the appropriate model
    if model_type == "pretrained":#组装成一个“前向生成流程”pipe
        pipe = StableDiffusionPipeline.from_pretrained(config.pretrained_model_path).to(device)
        unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_path, subfolder="unet").to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(config.pretrained_model_path).to(device)
        unet = UNet2DConditionModel.from_pretrained(config.fine_tuned_model_path).to(device)

    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_path, subfolder="scheduler")

    dataloader = prepare_test_dataloader(config)

    losses = []
    image_samples = []

    unet.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type} model"):
            clean_images = batch["pixel_values"].to(device)

            # Tokenize input poems
            inputs = tokenizer(batch["input_ids"], max_length=tokenizer.model_max_length,
                               padding="max_length", truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)

            # Forward pass
            latents = vae.encode(clean_images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            losses.append(loss.item())

            # Save some samples for visualization (first 3 samples)
            if len(image_samples) < 3:
                pipe.unet = unet
                generated_image = pipe(batch["input_ids"][0]).images[0]
                image_samples.append({
                    "original": clean_images[0].cpu(),
                    "generated": generated_image,
                    "poem": batch["input_ids"][0],
                    "image_id": batch["image_id"][0],
                    "loss": loss.item()
                })

    return losses, image_samples


def plot_comparison(pretrained_losses, finetuned_losses, pretrained_samples, finetuned_samples):
    # Create output directory
    os.makedirs("test_results", exist_ok=True)

    # Plot loss comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(pretrained_losses)), pretrained_losses, label='Pretrained', alpha=0.5)
    plt.plot(range(len(finetuned_losses)), finetuned_losses, label='Fine-tuned', alpha=0.5)
    plt.title("Test Loss Comparison")
    plt.xlabel("Index")
    plt.ylabel("MSE Loss")
    plt.legend()

    # Calculate and display average losses
    avg_pretrained = np.mean(pretrained_losses)
    avg_finetuned = np.mean(finetuned_losses)

    plt.subplot(1, 2, 2)
    plt.bar(['Pretrained', 'Fine-tuned'], [avg_pretrained, avg_finetuned], color=['blue', 'orange'])
    plt.title("Average Test Loss")
    plt.ylabel("MSE Loss")

    plt.tight_layout()
    plt.savefig("test_results/loss_comparison.png")
    plt.close()

    # Plot image comparisons
    num_samples = min(3, len(pretrained_samples))
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        original_img = pretrained_samples[i]["original"].permute(1, 2, 0).numpy()
        original_img = (original_img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        plt.imshow(original_img)
        plt.title(f"Original\n{pretrained_samples[i]['image_id']}")
        plt.axis('off')

        # Pretrained model output
        plt.subplot(1, 3, 2)
        plt.imshow(pretrained_samples[i]["generated"])
        plt.title(f"Pretrained\nLoss: {pretrained_samples[i]['loss']:.4f}\n{pretrained_samples[i]['poem']}")
        plt.axis('off')

        # Fine-tuned model output
        plt.subplot(1, 3, 3)
        plt.imshow(finetuned_samples[i]["generated"])
        plt.title(f"Fine-tuned\nLoss: {finetuned_samples[i]['loss']:.4f}\n{finetuned_samples[i]['poem']}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"test_results/image_comparison_{i}.png")
        plt.close()


def main():
    config = TestConfig()

    print("Evaluating pretrained model on images 100-120...")
    pretrained_losses, pretrained_samples = evaluate_model(config, "pretrained")

    print("\nEvaluating fine-tuned model on images 100-120...")
    finetuned_losses, finetuned_samples = evaluate_model(config, "finetuned")

    print("\nGenerating comparison plots...")
    plot_comparison(pretrained_losses, finetuned_losses, pretrained_samples, finetuned_samples)

    print("\nTest results (images 100-120):")
    print(f"Pretrained model average loss: {np.mean(pretrained_losses):.4f}")
    print(f"Fine-tuned model average loss: {np.mean(finetuned_losses):.4f}")
    print(f"Improvement: {((np.mean(pretrained_losses) - np.mean(finetuned_losses)) / np.mean(pretrained_losses) * 100):.2f} %")

    print("\nComparison images and loss plots saved to 'test_results' directory.")


if __name__ == "__main__":
    main()
