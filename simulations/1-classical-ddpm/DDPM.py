import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import time

    import deepinv
    import seaborn as sns
    import torch
    from torchvision import datasets, transforms

    return datasets, deepinv, sns, time, torch, transforms


@app.cell
def _():
    # device = "cpu"
    batch_size: int = 64
    image_size: int = 32
    return batch_size, image_size


@app.cell
def _(image_size: int, transforms):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ],
    )
    return (transform,)


@app.cell
def _(datasets, torch, transform):
    mnist = datasets.MNIST(
        root="./simulations/1/data",
        train=True,
        download=True,
        transform=transform,
    )
    dataset = torch.utils.data.Subset(mnist, list(range(0, len(mnist), 10)))
    dataset_len = len(dataset)
    return dataset, dataset_len


@app.cell
def _(batch_size: int, dataset, torch):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return (train_loader,)


@app.cell
def _():
    lr: float = 1e-4
    epochs: int = 4
    return epochs, lr


@app.cell
def _(deepinv, lr: float, torch):
    model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()
    return model, mse, optimizer


@app.cell
def _():
    beta_start: float = 1e-4
    beta_end: float = 0.02
    timesteps: int = 1000
    return beta_end, beta_start, timesteps


@app.cell
def _(beta_end: float, beta_start: float, timesteps: int, torch):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return (
        alphas,
        alphas_cumprod,
        betas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    )


@app.cell
def _(
    batch_size: int,
    dataset_len,
    mo,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    time,
    torch,
):
    def train(model, train_loader, optimizer, mse, epochs, timesteps):
        losses: list[float] = []

        for epoch in range(epochs):
            model.train()

            for i, (imgs, _) in enumerate(train_loader):
                noise = torch.randn_like(imgs)
                t = torch.randint(0, timesteps, (imgs.size(0),))

                noised_imgs = (
                    sqrt_alphas_cumprod[t, None, None, None] * imgs
                    + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
                )

                optimizer.zero_grad()
                estimated_noise = model.forward(noised_imgs, t, type_t="timestep")
                loss = mse(estimated_noise, noise).mean()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if loss.item() < 0.002:
                    mo.output.replace("Loss is low, stopping training.")
                    torch.save(
                        model.state_dict(),
                        "./simulations/1/trained_diffusion_model.pth",
                    )
                    return losses

                mo.output.clear()
                mo.output.append(f"Epoch: {epoch + 1}/{epochs}")
                mo.output.append(f"Progress: {(i * batch_size / dataset_len) * 100:.2f}%")
                mo.output.append(f"Loss: {loss.item():.4f}")

        torch.save(
            model.state_dict(),
            f"./simulations/1/trained_diffusion_model_{time.time():.0f}.pth",
        )

        return model, losses

    return (train,)


@app.cell
def _(epochs: int, model, mse, optimizer, timesteps: int, train, train_loader):
    train(model, train_loader, optimizer, mse, epochs, timesteps)
    return


@app.cell
def _(
    alphas,
    alphas_cumprod,
    betas,
    deepinv,
    image_size: int,
    mo,
    timesteps: int,
    torch,
):
    def sample(model, weights=None):
        if weights is not None:
            model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=weights)

        model.eval()

        n_samples = 32

        with torch.no_grad():
            x = torch.randn(n_samples, 1, image_size, image_size)

            for t in reversed(range(timesteps)):
                t_tensor = torch.ones(n_samples).long() * t

                predicted_noise = model(x, t_tensor, type_t="timestep")

                alpha = alphas[t]
                alpha_cumprod = alphas_cumprod[t]
                beta = betas[t]

                noise = torch.randn_like(x) if t > 0 else 0

                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
                ) + torch.sqrt(beta) * noise

                mo.output.replace(f"Sampling: {(timesteps - (t + 1)) * 100 / timesteps}%")

        return x

    return (sample,)


@app.cell
def _(model, sample):
    s = sample(model, weights="./simulations/1/trained_diffusion_model_1750610944.pth")
    return (s,)


@app.cell
def _(s, torch):
    x = torch.clamp(s, 0, 1)
    return (x,)


@app.cell
def _(plt, sns, x):
    def plot_images_seaborn_grid(x, rows=8, cols=4):
        x = x.squeeze(1)  # (32, 32, 32)
        _, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for idx, img in enumerate(x):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            sns.heatmap(img.cpu().numpy(), cmap="gray", cbar=False, ax=ax, square=True)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    plot_images_seaborn_grid(x, rows=8, cols=4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
