
import pandas as pd  # for data manipulation and analysis

from cloud_generator import run_wordcloud
from menus import get_available_filters, batch_choice, choose_filter_menu, options_menu 

import re  # for regular expression operations, used to filter strings
from functools import wraps  # for creating decorators, used to wrap the load_data function with text cleaning functionality


# Regular expressions
# re.compile is used to precompile the regex patterns for better performance when used repeatedly
_USERNAME_RE = r"@\w+"
_URL_RE = r"https?://t\.co/\S+|https?://\S+|www\.\S+"
_EXPLETIVES_RE = r"\b(?:fuck|shit|dick|asshole)\w*"

_CLEAN_RE = re.compile(
    rf"(?:{_URL_RE}|{_USERNAME_RE}|{_EXPLETIVES_RE})",
    re.IGNORECASE,
)

def _clean_one(x):
    # Leave non-strings unchanged 
    if not isinstance(x, str):
        return x
    return _CLEAN_RE.sub("", x)

def clean_text_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        text_data = func(*args, **kwargs)

        try:
            if isinstance(text_data, pd.Series):
                return text_data.map(_clean_one)
        except Exception:
            pass

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


def main():
    """
    Main function to run the word cloud generation process.
    Steps:
        1. Load the CSV file and extract the 'Network' and 'Content' columns.
        2. Get the unique filters from the 'Network' column.
        3. Prompt the user to choose between batch mode and interactive mode.
        4. If batch_mode is True, generate word clouds for all filters in batch mode.
           Otherwise, run in interactive mode to select a filter and generate the corresponding word cloud.
        5. In interactive mode, also allow the user to configure word cloud options before generating the word cloud.
    """
    file_path = "data.csv"
    mask_image_path = None

    df = pd.read_csv(file_path, usecols=["Network", "Content"])

    filters = get_available_filters(df)

    batch_mode = batch_choice()

    if batch_mode:
        # batch mode
        mask_image_path = None
        for f in filters:
            text_data = load_data(df, f)
            run_wordcloud(text_data, f, mask_image_path, show=False)

        # also generate the "all" cloud
        text_data = load_data(df, network_filter==None)
        run_wordcloud(df, None, mask_image_path, show=False)

    else:
        # interactive mode
        print(f"mask image path: {mask_image_path}")
        network_filter = choose_filter_menu(filters, prechosen_number=None)
        text_data = load_data(df, network_filter)
        configuration_options = (
            {  # default options for interactive mode, can be adjusted as needed
                "max_words": 200,
                "prefer_horizontal": 0.8,
            }
        )
        prompt = "Configure word cloud options:"
        options = options_menu(configuration_options, prompt)
        run_wordcloud(text_data, network_filter, mask_image_path, show=True, options=options)


if __name__ == "__main__": 
    main()