import numpy as np  # for array manipulation like handling image data for masks
import pandas as pd  # for typing of text data
from PIL import Image  # for image processing
from wordcloud import WordCloud, STOPWORDS  # for generating the word cloud and using built-in stopwords to filter out common words



def build_tight_mask(
    mask_image_path: str, out_size: int = 1800, pad: int = 50
) -> np.ndarray:
    """
    Build a tight mask for the word cloud based on the alpha channel of the input image.
    Args:
        mask_image_path: The file path to the input image to use as a mask.
        out_size: The desired output size (width and height) of the mask in pixels (default: 1600).
        pad: The number of pixels to pad around the detected drawable area to ensure words fit well (default: 50).
    Returns:
        A 2D numpy array representing the mask, where 255 indicates drawable areas and 0 indicates blocked areas.
    """
    img = Image.open(mask_image_path).convert("RGBA")
    arr = np.array(img)

    # Use alpha channel directly
    alpha = arr[:, :, 3]

    # WordCloud rule:
    # 0 = blocked
    # 255 = drawable
    mask = np.where(alpha > 10, 255, 0).astype(np.uint8)

    # ---- Crop to remove empty margins ----
    ys, xs = np.where(mask == 255)

    if len(xs) == 0:
        raise ValueError("Mask contains no drawable region.")

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)

    mask = mask[y0 : y1 + 1, x0 : x1 + 1]

    # Resize upward
    mask = Image.fromarray(mask).resize((out_size, out_size), Image.NEAREST)

    return np.array(mask)


def solid_black_color_func(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    """
    Color function that returns a solid black color.
    Returns:
        A hex color string representing black.
    """
    return "#000000"


def generate_word_cloud(text_data, mask_image_path=None, options: dict | None = None):
    """
    Generate a word cloud from the provided text data, optionally using a mask image and configuration options.
    Args:
        text_data: A list or pandas Series of strings to generate the word cloud from.
        mask_image_path: Optional file path to an image to use as a mask for the word cloud.
        options: Optional dictionary of configuration options for the word cloud generation (e.g., max_words, prefer_horizontal).
    Returns:
        A WordCloud object representing the generated word cloud.
    """
    options = options or {}
    text = " ".join(text_data)

    # base wordcloud options (shared for both masked and unmasked)
    wc_kwargs = {
        "stopwords": STOPWORDS,  # use default stopwords to filter out common words
        "background_color": "white",  # set background to white for better contrast with black words
        "contour_width": 0,  # remove black outlines
        "scale": 2,  # increase resolution for better quality
        "margin": 1,  # smaller margins between words
        "relative_scaling": 1.0,  # more emphasis on word frequency
        "max_font_size": None,  # allow font size to scale with frequency
        "min_font_size": 4,  # set a minimum font size to ensure small words are visible
        "max_words": int(
            options.get("max_words", 200)
        ),  # limit the number of words to fit better in the mask
        "prefer_horizontal": options.get(
            "prefer_horizontal", 0.8
        ),  # prefer horizontal words to better fill the mask shape
        "collocations": False,  # reduces duplicate bigrams taking space
        "random_state": 42,  # set a fixed random state for reproducibility
    }

    if mask_image_path:
        wc_kwargs["mask"] = build_tight_mask(mask_image_path, out_size=1800, pad=50)

    wc = WordCloud(**wc_kwargs).generate(text)

    if mask_image_path:
        wc = wc.recolor(color_func=solid_black_color_func)

    return wc


def save_word_cloud(wordcloud, output_path="wordcloud.png"):
    """
    Save the generated word cloud to a file with no borders or extra metadata.
    Args:
        wordcloud: The WordCloud object to save.
        output_path: The file path to save the word cloud image (default: "wordcloud.png").
    """
    wordcloud.to_file(output_path)


def sanitize_tag(tag: str) -> str:
    """
    Sanitize a string to be used as a filename tag by replacing non-alphanumeric characters with underscores.
    Args:
        tag: The input string to sanitize.
    Returns:
        A sanitized string suitable for use in filenames.
    """
    return "".join(
        c if c.isalnum() or c in ("_", "-") else "_" for c in tag.strip().lower()
    ).strip("_")


def display_word_cloud(wordcloud):
    """
    Display the generated word cloud using matplotlib.
    Args:
        wordcloud: The WordCloud object to display.
    """
    import matplotlib.pyplot as plt

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def run_wordcloud(text_data:pd.Series, network_filter, mask_image_path=None, show=False, options=None):
    """
    Run the word cloud generation process for a specific network filter.
    Args:
        text_data: A pandas Series containing the text data to generate the word cloud from.
        network_filter: An optional string to filter the 'Network' column. If None, no filtering is applied.
        mask_image_path: Optional file path to an image to use as a mask for the word cloud.
        show: If True, display the generated word cloud after saving it. 
        options: Optional dictionary of configuration options for the word cloud generation (e.g., max_words, prefer_horizontal).
    """

    wordcloud = generate_word_cloud(text_data, mask_image_path, options)

    tag = "all" if network_filter is None else sanitize_tag(network_filter)
    output_filename = f"wordcloud_{tag}.png"

    save_word_cloud(wordcloud, output_filename)
    print(f"Saved: {output_filename}")

    if show:
        print(f"Displayed word cloud for filter: {network_filter}")
        display_word_cloud(wordcloud)
