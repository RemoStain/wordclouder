from getpass import getpass
def safe_input(expected_type, message=None, default=None, is_password:bool=False):
    """
    Prompt the user for input and safely convert it to the given type.

    Args:
        expected_type (type): The type to convert the input into (e.g., int, float, str, bool).
        message (str, optional): Custom message to display instead of the default.
        default (any, optional): Default value to return if conversion fails.
        is_password (bool, optional): If True, input will be hidden (for passwords).

    Returns:
        The converted value of the correct type, or None if interrupted.
    """
    while True:
        try:
            prompt = (
                message
                if message is not None
                else f"Enter a(n) ({expected_type.__name__}): "
            )
            if is_password:
                user_input = getpass(prompt)
            else:
                user_input = input(prompt)

            # Special handling for bool
            if expected_type is bool:
                lowered = user_input.strip().lower()
                if lowered in ("true", "1", "yes", "y"):
                    return True
                elif lowered in ("false", "0", "no", "n"):
                    return False
                else:
                    raise ValueError("Invalid boolean input")

            if expected_type is str:
                if user_input == "" and default is not None:
                    return default

            # Attempt conversion for other types
            return expected_type(user_input)

        except ValueError:
            if default is not None:
                if user_input != "":
                    print(f"Invalid input. Using default value: {default}")
                return default
            print(f"Invalid input. Please enter a valid {expected_type.__name__}.")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Aborting input.")
            return None  # or raise SystemExit if you want to exit the program
        except EOFError:
            print("\nEnd of input detected (Ctrl+D). Returning None.")
            return None

if __name__ == "__main__":
    # Example usage
    age = safe_input(int, "Enter your age: ", default=18)
    print(f"Your age is: {age}")
    height = safe_input(float, "Enter your height in meters: ", default=1.75)
    print(f"Your height is: {height} meters")
    name = safe_input(str, "Enter your name: ", default="Guest")
    print(f"Hello, {name}!")
    wants_newsletter = safe_input(bool, "Do you want to subscribe to the newsletter? (yes/no): ", default=False)
    print(f"Newsletter subscription: {wants_newsletter}")
    password = safe_input(str, "Enter your password: ", is_password=True)
    print(f"Your password is: {'*' * len(password) if password else 'None'}")