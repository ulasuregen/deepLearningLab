"""
Plotting utilities for the colorization training loop.
"""

import os
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.color import lab2rgb
from tqdm import tqdm


def plot_losses(train_losses, val_losses):
    """
    Plot training vs validation loss curves, updating in place (useful for
    calling once per epoch inside a training loop in a Jupyter notebook).

    Args:
        train_losses (list[float]): average training loss per epoch so far
        val_losses (list[float]): average validation loss per epoch so far
    """
    clear_output(wait=True)  # clears previous output before redrawing

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val loss", linewidth=2, linestyle="--")

    best_epoch = val_losses.index(min(val_losses)) + 1
    plt.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.5,
                label=f"Best val loss (epoch {best_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_predictions(model, dataloader, device, n=4):
    """
    Run a batch through the model and display grayscale input, predicted
    color, and ground truth color side by side.

    Args:
        model: trained (or in-progress) ColorizationUNet
        dataloader: typically val_loader, so predictions reflect unseen data
        device: torch.device the model/tensors live on (e.g. "mps", "cuda", "cpu")
        n (int): number of samples from the batch to display
    """
    model.eval()
    L_batch, ab_batch = next(iter(dataloader))
    L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

    with torch.no_grad():
        pred_ab = model(L_batch)

    # Move everything back to CPU + numpy for plotting
    L_batch = L_batch.cpu().numpy()
    ab_batch = ab_batch.cpu().numpy()
    pred_ab = pred_ab.cpu().numpy()

    n = min(n, L_batch.shape[0])  # don't exceed the actual batch size
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]  # keep indexing consistent for a single row

    for i in range(n):
        # Un-normalize L: (L_norm + 1) * 50  ->  back to [0, 100]
        L = (L_batch[i, 0] + 1) * 50

        # Un-normalize ab: ab_norm * 110  ->  back to roughly [-110, 110]
        ab_real = ab_batch[i].transpose(1, 2, 0) * 110
        ab_pred = pred_ab[i].transpose(1, 2, 0) * 110

        # Stitch L + ab together into full Lab images
        lab_real = np.concatenate([L[:, :, None], ab_real], axis=2)
        lab_pred = np.concatenate([L[:, :, None], ab_pred], axis=2)

        rgb_real = lab2rgb(lab_real)
        rgb_pred = lab2rgb(lab_pred)

        axes[i, 0].imshow(L, cmap='gray')
        axes[i, 0].set_title("Grayscale Input")
        axes[i, 1].imshow(rgb_pred)
        axes[i, 1].set_title("Predicted Color")
        axes[i, 2].imshow(rgb_real)
        axes[i, 2].set_title("Ground Truth")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def download_pexels_images(queries, target_total, save_dir,
                            per_page=80, pages_per_query=5, api_key=None):
    """
    Search Pexels for a list of query terms and download images until
    target_total is reached (or all queries are exhausted).

    Args:
        queries (list[str]): search terms, e.g. ["forest", "pine forest"]
        target_total (int): stop once this many images have been collected
        save_dir (str): folder to save downloaded images into (created if missing)
        per_page (int): images per API page request (Pexels max is 80)
        pages_per_query (int): how many pages to request per query before
            moving to the next query
        api_key (str, optional): Pexels API key. Defaults to the
            PEXELS_API_KEY environment variable if not provided.

    Returns:
        (int, int): (number of images successfully downloaded, number failed)
    """
    api_key = api_key or os.environ.get('PEXELS_API_KEY')
    if not api_key:
        raise ValueError(
            "No Pexels API key found. Pass api_key=... or set the "
            "PEXELS_API_KEY environment variable."
        )

    headers = {"Authorization": api_key}
    url = 'https://api.pexels.com/v1/search'
    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: Collect image URLs across queries + pages ---
    img_urls = []

    for query in queries:
        for page in range(1, pages_per_query + 1):
            if len(img_urls) >= target_total:
                break

            params = {'query': query, 'per_page': per_page, 'page': page}
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 429:
                print("Rate limited by Pexels — stopping URL collection early.")
                break
            if response.status_code != 200:
                print(f"Skipping {query} page {page}: status {response.status_code}")
                continue

            data = response.json()
            photos = data.get('photos', [])
            if not photos:
                break  # no more results for this query

            for photo in photos:
                img_urls.append(photo['src']['large'])

        if len(img_urls) >= target_total:
            break

    img_urls = img_urls[:target_total]
    print(f"Collected {len(img_urls)} image URLs.")

    # --- Step 2: Download images ---
    failed = 0
    for i, img_url in enumerate(tqdm(img_urls, desc="Downloading")):
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(f'{save_dir}/image_{i}.jpg', 'wb') as f:
                    f.write(response.content)
            else:
                failed += 1
        except requests.exceptions.RequestException:
            failed += 1

    downloaded = len(img_urls) - failed
    print(f"Done. {downloaded} downloaded, {failed} failed.")
    return downloaded, failed


def show_vae_reconstructions(model, dataloader, device, n=8):
    """
    Run a batch through the VAE and display original vs. reconstructed
    digits side by side.

    Args:
        model: trained (or in-progress) VAE
        dataloader: typically val_loader, so reconstructions reflect unseen data
        device: torch.device the model/tensors live on (e.g. "mps", "cuda", "cpu")
        n (int): number of samples from the batch to display
    """
    model.eval()
    x, _ = next(iter(dataloader))
    x = x.to(device)

    with torch.no_grad():
        x_hat, mu, logvar = model(x)

    x = x.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

    n = min(n, x.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3))

    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(x_hat[i, 0], cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.tight_layout()
    plt.show()


def show_vae_samples(model, latent_dim, device, n=8):
    """
    Sample z ~ N(0, 1) and decode to see what the VAE generates from
    scratch (no input image — tests the quality of the learned latent space).

    Args:
        model: trained VAE
        latent_dim (int): dimensionality of the latent space
        device: torch.device
        n (int): number of samples to generate
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        samples = model.Decoder(z).cpu().numpy()

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 1.5))
    for i in range(n):
        axes[i].imshow(samples[i, 0], cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()