
from safe_input import safe_input

def get_available_filters(df) -> list[str]:
    """
    Get the unique non-null values from the 'Network' column of the DataFrame.
    Args:
        df: A pandas DataFrame containing a 'Network' column.
    Returns:
        A sorted list of unique non-null values from the 'Network' column as strings.
        
    """
    return sorted(set(df["Network"].dropna().astype(str).unique()))

def batch_choice() -> bool:
    """
    Prompt the user to choose between batch mode and interactive mode.
    Returns:
        True if batch mode is selected, False if interactive mode is selected.
    """
    try:
        while True:
            choice = safe_input(
                str,
                "Select mode: \n(1) Batch Mode (generate all word clouds at once), \n(2) Interactive Mode (generate one selected cloud) \n[default = 2]: ",
                "2",
            ).strip().lower()
            if choice in ("1", "batch", "b"):
                return True
            elif choice in ("2", "interactive", "i"):
                return False
            else:
                print("Invalid selection. Please enter '1' for Batch Mode or '2' for Interactive Mode.")
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt.")
        raise SystemExit(0)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(0)
        

def choose_filter_menu(filters: set[str], prechosen_number) -> str | None:
    """
    Display a menu to choose a network filter from the available options.
    Args:
        filters: A set of available network filter options.
        prechosen_number: An optional integer representing a prechosen filter option.
            If provided and valid, the corresponding filter will be returned without prompting the user.
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
