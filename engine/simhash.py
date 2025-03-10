import hashlib


def compute_word_frequencies(token_list):
    """
    Computes the frequency of each unique token in the given list.

    Args:
        token_list (list): A list of tokens.

    Returns:
        dict: A dictionary mapping tokens to their respective frequencies.
    """
    frequency_dict = dict()

    for token in token_list:
        if token not in frequency_dict.keys():
            frequency_dict[token] = 1
        else:
            frequency_dict[token] += 1

    return frequency_dict



def extract_features(tokens):
    """
    Gets the features or values for the tokens by frequency

    Args:
        tokens (list): A list of tokens.

    Returns:
        list: The list of featured tokens
    """
    # Get word frequencies from tokens
    frequencies = compute_word_frequencies(tokens)

    # Create a feature for each word based on its frequency
    features = []
    for word, freq in frequencies.items():
        # Repeat the word based on its frequency
        features.extend([word] * freq)

    return features

def simhash(features, bit_length=64):
    """
    Computes a SimHash value for a given set of features.

    Args:
        features (list): A list of feature tokens.
        bit_length (int): The length of the hash in bits (default is 64).

    Returns:
        int: The computed SimHash value.
    """
    v = [0] * bit_length

    for feature in features:
        h = int(hashlib.sha1(feature.encode('utf-8')).hexdigest(), 16)  # Convert feature to hash
        for i in range(bit_length):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    # Convert vector to binary hash
    return sum(1 << i for i in range(bit_length) if v[i] > 0)


def hamming_distance(hash1, hash2):
    """
    Computes the Hamming distance between two SimHash values.

    Args:
        hash1 (int): First hash value.
        hash2 (int): Second hash value.

    Returns:
        int: The Hamming distance between the two hashes.
    """
    x = (hash1 ^ hash2) & ((1 << 64) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ans

# Set to store all simhashes
previous_hashes = set()


def is_same_content(tokens, threshold=2):
    """
    Checks if the hashes are similar to previously seen pages.

    Args:
        tokens (list): A list of tokenized words from the content.
        threshold (int, optional): The maximum Hamming distance allowed for content to be considered similar. Defaults to 3.

    Returns:
        bool: True if the content is similar to previous pages, False otherwise.
    """
    global previous_hashes
    current_hash = simhash(extract_features(tokens))

    if not previous_hashes:
        previous_hashes.add(current_hash)
        return False

    smallest_dist = float('inf')

    for old_hash in previous_hashes:
        dist = hamming_distance(old_hash, current_hash)
        smallest_dist = min(smallest_dist, dist)

    print(f'DISTANCE: {smallest_dist}')  # Debugging info

    if smallest_dist <= threshold:
        return True  # Similar content detected

    previous_hashes.add(current_hash)
    return False