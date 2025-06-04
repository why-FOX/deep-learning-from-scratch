import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm


# 配置参数
class TrainingConfig:
    model_path = "D:\\github\\deep-learning-from-scratch\\co-stable diffusion\\stable-diffusion-v1-5\\stable-diffusion-v1-5"
    image_dir = os.path.join(model_path, "images", "images")
    csv_path = os.path.join(model_path, "POEM_IMAGE.csv")

    batch_size = 1
    num_epochs = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = "no"  # 不使用 fp16

    output_dir = "fine_tuned_model_cpu"
    logging_dir = "logs_cpu"
    save_model_epochs = 1
    save_images_epochs = 1

    resolution = 512
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    max_samples = 100  # 减少样本数，加速调试

#自定义dataset类
class PoemImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path, sep='\t', encoding='utf-8-sig')
        if max_samples is not None:
            self.df = self.df.head(max_samples)
        #取定义的头几个数据
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((TrainingConfig.resolution, TrainingConfig.resolution)),
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
        return {"pixel_values": image, "input_ids": poem}

#存入数据加载器
def prepare_dataloaders(config):
    dataset = PoemImageDataset(
        csv_path=config.csv_path,
        image_dir=config.image_dir,
        transform=transforms.Compose([
            transforms.Resize((config.resolution, config.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        max_samples=config.max_samples
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)


def train_model(config):
    device = torch.device("cpu")

    # 加载模型和模块到 CPU（无 dtype 设置）
    pipe = StableDiffusionPipeline.from_pretrained(config.model_path).to(device)
    unet = UNet2DConditionModel.from_pretrained(config.model_path, subfolder="unet").to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_path, subfolder="scheduler")

    dataloader = prepare_dataloaders(config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, config.lr_warmup_steps,
                                                   config.num_epochs * len(dataloader))

    loss_history = []
    global_step = 0

    for epoch in range(config.num_epochs):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            clean_images = batch["pixel_values"].to(device)

            inputs = tokenizer(batch["input_ids"], max_length=tokenizer.model_max_length,
                               padding="max_length", truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            #拿到tensor input
            # 训练前向传播（无混合精度）
            latents = vae.encode(clean_images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            loss_history.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

        # 保存模型
        save_path = os.path.join(config.output_dir, f"model_epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        unet.save_pretrained(os.path.join(save_path, "unet"))
        noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))

        # 推理生成图像
        pipe.unet = unet.to(device)
        test_prompts = [
            "举头望明月,低头思故乡, Chinese ink painting, traditional style",
            "白日依山尽,黄河入海流, Chinese ink painting, traditional style"
        ]
        for prompt in test_prompts:
            image = pipe(prompt).images[0]
            image.save(os.path.join(config.output_dir, f"epoch_{epoch}_{prompt[:20]}.png"))

    # 画损失图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(config.output_dir, "training_loss.png"))
    plt.close()

    return unet


def main():
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)

    print("Training on CPU...")
    trained_unet = train_model(config)
    print("Training completed!")

    final_save_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    trained_unet.save_pretrained(os.path.join(final_save_path, "unet"))
    print(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()






'''
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm


# 配置参数
class TrainingConfig:
    model_path = "D:\\github\\deep-learning-from-scratch\\co-stable diffusion\\stable-diffusion-v1-5\\stable-diffusion-v1-5"
    image_dir = os.path.join(model_path, "images", "images")
    csv_path = os.path.join(model_path, "POEM_IMAGE.csv")

    batch_size = 1  # 小批次以减少显存占用
    num_epochs = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # 使用混合精度

    output_dir = "fine_tuned_model_gpu"
    logging_dir = "logs_gpu"
    save_model_epochs = 5
    save_images_epochs = 5

    resolution = 512
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    max_samples = 100  # 只训练前100个样本


class PoemImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path, sep='\t', encoding='utf-8-sig')
        if max_samples is not None:
            self.df = self.df.head(max_samples)

        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((TrainingConfig.resolution, TrainingConfig.resolution)),
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

        poem = row['poem']
        return {"pixel_values": image, "input_ids": poem}


def prepare_dataloaders(config):
    dataset = PoemImageDataset(
        csv_path=config.csv_path,
        image_dir=config.image_dir,
        transform=transforms.Compose([
            transforms.Resize((config.resolution, config.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        max_samples=config.max_samples
    )

    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)


def train_model(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if config.mixed_precision == "fp16" and use_cuda else torch.float32

    # 只用于推理的 pipeline，使用 float32 加载即可
    pipe = StableDiffusionPipeline.from_pretrained(config.model_path).to(device)

    # 加载核心模块（参与训练的部分）
    unet = UNet2DConditionModel.from_pretrained(config.model_path, subfolder="unet").to(device, dtype=dtype)
    vae = pipe.vae.to(device, dtype=dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_path, subfolder="scheduler")

    dataloader = prepare_dataloaders(config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, config.lr_warmup_steps,
                                                   config.num_epochs * len(dataloader))

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))  # Mixed precision scaler
    loss_history = []
    global_step = 0

    for epoch in range(config.num_epochs):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            clean_images = batch["pixel_values"].to(device, dtype=dtype)

            # Tokenize input poems
            inputs = tokenizer(batch["input_ids"], max_length=tokenizer.model_max_length,
                               padding="max_length", truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)

            with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
                # Encode image -> latents
                latents = vae.encode(clean_images).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # backward + optimizer
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            loss_history.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

        # Save model
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            save_path = os.path.join(config.output_dir, f"model_epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            unet.save_pretrained(os.path.join(save_path, "unet"))
            noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))

        # Save generated images
        if (epoch + 1) % config.save_images_epochs == 0 or epoch == config.num_epochs - 1:
            pipe.unet = unet.to(device, dtype=torch.float16 if use_cuda else torch.float32)  # update unet
            test_prompts = [
                "举头望明月,低头思故乡, Chinese ink painting, traditional style",
                "白日依山尽,黄河入海流, Chinese ink painting, traditional style"
            ]
            for prompt in test_prompts:
                with torch.autocast("cuda", enabled=(dtype == torch.float16)):
                    image = pipe(prompt).images[0]
                image.save(os.path.join(config.output_dir, f"epoch_{epoch}_{prompt[:20]}.png"))

    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(config.output_dir, "training_loss.png"))
    plt.close()

    return unet



def main():
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)

    print("Training on GPU using 100 samples with memory optimization...")
    trained_unet = train_model(config)
    print("Training completed!")

    final_save_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    trained_unet.save_pretrained(os.path.join(final_save_path, "unet"))
    print(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
'''