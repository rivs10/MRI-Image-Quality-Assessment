
import cv2
import numpy as np
from skimage.measure import shannon_entropy


def add_gaussian_noise(image, sigma, rng=None):
    """Add Gaussian noise to a grayscale image."""
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(image, kernel_size):
    """Apply Gaussian blur to simulate loss of sharpness."""
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), 0
    )


def degrade_image(image, quality, rng=None):
    """
    Generate simulated quality levels:
    0 = Good, 1 = Moderate, 2 = Poor.
    """
    if quality == 0:
        return image.copy()

    if quality == 1:
        image = add_gaussian_noise(image, 15, rng)
        return add_blur(image, 3)

    if quality == 2:
        image = add_gaussian_noise(image, 35, rng)
        return add_blur(image, 7)

    raise ValueError("quality must be 0, 1, or 2")


def extract_features(image):
    """Extract quantitative features from a grayscale image."""
    image = np.asarray(image)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = np.clip(image, 0, 255).astype(np.uint8)
    img = image.astype(np.float32) / 255.0

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx**2 + gy**2)

    edges = cv2.Canny(image, 100, 200)

    return {
        "mean_intensity": float(np.mean(img)),
        "std_intensity": float(np.std(img)),
        "median_intensity": float(np.median(img)),
        "contrast": float(
            np.percentile(img, 95) - np.percentile(img, 5)
        ),
        "entropy": float(shannon_entropy(image)),
        "sharpness": float(
            cv2.Laplacian(img, cv2.CV_32F).var()
        ),
        "edge_density": float(np.mean(edges > 0)),
        "mean_gradient": float(np.mean(gradient)),
        "p10": float(np.percentile(img, 10)),
        "p25": float(np.percentile(img, 25)),
        "p75": float(np.percentile(img, 75)),
        "p90": float(np.percentile(img, 90)),
    }
