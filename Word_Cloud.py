import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

from safe_input import safe_input

import re
from functools import wraps

# Regular expressions to identify usernames and URLs in the text
_USERNAME_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://t\.co/\S+|https?://\S+|www\.\S+")


def clean_text_decorator(func):
    """
    Decorator to clean text data by removing URLs and usernames from the 'Content' column.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        text_data = func(*args, **kwargs)  # df/network_filter stay intact

        # If it's a pandas Series, preserve the type/index
        try:
            import pandas as pd

            if isinstance(text_data, pd.Series):
                return text_data.map(lambda s: _URL_RE.sub("", _USERNAME_RE.sub("", s)))
        except Exception:
            pass

        # Generic iterable path (list/tuple/etc.)
        return [_URL_RE.sub("", _USERNAME_RE.sub("", text)) for text in text_data]

    return wrapper


@clean_text_decorator
def load_data(df, network_filter=None):
    """
    Load the 'Content' column from the DataFrame, optionally filtering by the 'Network' column.
    Args:
        df: A pandas DataFrame containing 'Network' and 'Content' columns.
        network_filter: An optional string to filter the 'Network' column. If None, no filtering is applied.
    Returns:
        A pandas Series containing the 'Content' data, filtered by the specified network if provided.
    """
    if network_filter is not None:
        df = df[df["Network"] == network_filter]
    return df["Content"].dropna().astype(str)


def get_available_filters(df) -> set[str]:
    """
    Get the unique non-null values from the 'Network' column of the DataFrame.
    Args:
        df: A pandas DataFrame containing a 'Network' column.
    Returns:
        A set of unique non-null values from the 'Network' column as strings.
    """
    return set(df["Network"].dropna().astype(str).unique())


def choose_filter_menu(filters: set[str], prechosen_number) -> str | None:
    """
    Display a menu to choose a network filter from the available options.
    Args:
        filters: A set of available network filter options.
    Returns:
        The selected network filter as a string, or None if "all" is selected.
    """
    options = ["all"] + sorted(filters, key=lambda s: s.lower())
    # If a prechosen number is provided and valid, return the corresponding filter without prompting the user
    if prechosen_number is not None and 1 <= prechosen_number <= len(options):
        selected = options[prechosen_number - 1]
        return None if selected == "all" else selected

    while True:
        print("\nSelect a network filter:")
        for i, opt in enumerate(options, start=1):
            print(f"{i}. {opt}")
        print("0. Exit")

        choice = safe_input(int, "Enter choice number (default = 1): ", 1)
        if choice == 0:
            raise SystemExit(0)

        try:
            idx = choice - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                return None if selected == "all" else selected
        except ValueError:
            pass

        print("Invalid selection. Try again.")


def options_menu(options: dict, prompt: str) -> dict:
    """
    Display the options for the wordcloud generation and return the users responses as a dict

    Args:
        options: A dict of options and defaults to display to the user.
        prompt: A string prompt to display before the options.
    Returns:
        A dictionary containing the user's responses for each option, where the keys are the option names and the values are the corresponding number responses.
    """
    print(prompt)
    responses = {}
    for opt, default in options.items():
        response = safe_input(
            float,
            f"Enter value for {opt} (default = {default}): ",
            default,
        )
        responses[opt] = response
    return responses


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


def run_wordcloud(df, network_filter, mask_image_path=None, show=False, options=None):
    """
    Run the word cloud generation process for a specific network filter.
    Args:
        df: A pandas DataFrame containing 'Network' and 'Content' columns.
        network_filter: An optional string to filter the 'Network' column. If None, no filtering is applied.
        mask_image_path: Optional file path to an image to use as a mask for the word cloud.
        show: If True, display the generated word cloud after saving it.
    """
    text_data = load_data(df, network_filter)

    wordcloud = generate_word_cloud(text_data, mask_image_path, options)

    tag = "all" if network_filter is None else sanitize_tag(network_filter)
    output_filename = f"wordcloud_{tag}.png"

    save_word_cloud(wordcloud, output_filename)
    print(f"Saved: {output_filename}")

    if show:
        print(f"Displayed word cloud for filter: {network_filter}")
        display_word_cloud(wordcloud)


def main(batch_mode: bool):
    """
    Main function to run the word cloud generation process.
    Steps:
        1. Load the CSV file and extract the 'Network' and 'Content' columns.
        2. Get the unique filters from the 'Network' column.
        3. If batch_mode is True, generate word clouds for all filters in batch mode.
           Otherwise, run in interactive mode to select a filter and generate the corresponding word cloud.

    Args:
        batch_mode: If True, generates word clouds for all filters in batch mode. If False, runs in interactive mode to select a filter.
    """
    file_path = "data.csv"
    mask_image_path = None

    df = pd.read_csv(file_path, usecols=["Network", "Content"])
    filters = sorted(get_available_filters(df))

    if batch_mode:
        # batch mode
        mask_image_path = None
        for f in filters:
            run_wordcloud(df, f, mask_image_path, show=False)

        # also generate the "all" cloud
        run_wordcloud(df, None, mask_image_path, show=False)

    else:
        # interactive mode
        print(f"mask image path: {mask_image_path}")
        network_filter = choose_filter_menu(filters, prechosen_number=None)
        configuration_options = (
            {  # default options for interactive mode, can be adjusted as needed
                "max_words": 200,
                "prefer_horizontal": 0.8,
            }
        )
        prompt = "Configure word cloud options:"
        options = options_menu(configuration_options, prompt)
        run_wordcloud(df, network_filter, mask_image_path, show=True, options=options)


if __name__ == "__main__":
    batch_mode = False  # Set to True for batch mode, False for interactive mode
    main(batch_mode)
