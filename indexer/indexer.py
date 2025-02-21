import re

def tokenize(text):
    """
        Tokenizes the given text into words, preserving alphanumeric sequences and apostrophes.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list: A list of tokens extracted from the text.
    """
    # Split by non-word characters and convert to lowercase
    text_tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return text_tokens


