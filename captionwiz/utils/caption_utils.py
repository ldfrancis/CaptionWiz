def post_process(caption: str) -> str:
    """Use only the part of the generated caption before the end of sentence token <end>

    Args:
        capiton (str): The generated caption

    Returns:
        (str): The sentence from the caption, upto <end>
    """

    end_of_sentence = "<end>"
    sentence = caption.split(end_of_sentence)[0]

    return sentence
