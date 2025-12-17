import re
from pathlib import Path

import nltk


def print_nltk_pos_tags():
    with open(path_to_hs_file, "r", encoding="utf-8") as hs_file:
        hs_file_contents = hs_file.readlines()

        for i, sentence in enumerate(hs_file_contents):
            print(i)
            print("  ", nltk.tag.pos_tag(nltk.tokenize.word_tokenize(sentence)))


def apply_normalization_rules():
    normalization_rules = {
        r"\sye\s": r" you ",
        r"bre\s": r"ber ",
        r"bre$": r"ber ",
        r"þ": r"th",
        r"y([bcdfghjknmlnpqrstvwxz])": r"i\1",
        r"ffor": r"for",
        r"\swn": r" un",
        r"\svn": r" un",
        r"sch": r"sh",
        r"eth\s": r"e ",
        r"ie\s": r"y ",
    }

    with open(path_to_hs_file, "r", encoding="utf-8") as hs_file:
        hs_file_contents = hs_file.readlines()

    hs_file_contents_rules_applied = []
    for sentence in hs_file_contents[200:]:
        for regex, substitution in normalization_rules.items():
            sentence = re.sub(regex, substitution, sentence)

        hs_file_contents_rules_applied.append(sentence)

    with open(
        Path(path_to_assignment_4, "icamet-samples_hs_rules.txt"), "w", encoding="utf-8"
    ) as hs_file_with_rules:
        hs_file_with_rules.write("".join(hs_file_contents_rules_applied))


def get_normalization_accuracy(
    historical_text: list[str], ground_truth: list[str]
) -> float:
    if len(historical_text) != len(ground_truth):
        raise ValueError(
            "Number of sentences in the historical text and the ground truth text should be the same!"
        )

    tokenized_historical_text = []
    for sentence in historical_text:
        tokenized_historical_text.append(nltk.tokenize.word_tokenize(sentence))

    tokenized_ground_truth = []
    for sentence in ground_truth:
        tokenized_ground_truth.append(nltk.tokenize.word_tokenize(sentence))

    total_token_count = 0
    matching_token_count = 0
    for historical_sentence, ground_truth_sentence in zip(
        tokenized_historical_text, tokenized_ground_truth
    ):
        for historical_token, ground_truth_token in zip(
            historical_sentence, ground_truth_sentence
        ):
            if historical_token == ground_truth_token:
                matching_token_count += 1

            total_token_count += 1

    return round(matching_token_count / total_token_count * 100, 1)


def get_pos_tag_similarity(
    historical_text: list[str], normalized_text: list[str]
) -> float:
    if len(historical_text) != len(normalized_text):
        raise ValueError(
            "Number of sentences in the historical text and the normalized text should be the same!"
        )

    tagged_historical_text = []
    for sentence in historical_text:
        tagged_historical_text.append(
            nltk.tag.pos_tag(nltk.tokenize.word_tokenize(sentence))
        )

    tagged_normalized_text = []
    for sentence in normalized_text:
        tagged_normalized_text.append(
            nltk.tag.pos_tag(nltk.tokenize.word_tokenize(sentence))
        )

    total_token_count = 0
    matching_tag_count = 0
    for historical_sentence, normalized_sentence in zip(
        tagged_historical_text, tagged_normalized_text
    ):
        for historical_tag, normalized_tag in zip(
            historical_sentence, normalized_sentence
        ):
            if historical_tag[1] == normalized_tag[1]:
                matching_tag_count += 1
            else:
                print(f"Before: {historical_tag}; After: {normalized_tag}")

            total_token_count += 1

    return round(matching_tag_count / total_token_count * 100, 1)


if __name__ == "__main__":
    path_to_assignment_4 = Path(__file__).parents[0]
    path_to_hs_file = Path(path_to_assignment_4, "icamet-samples_hs.txt")
    path_to_en_file = Path(path_to_assignment_4, "icamet-samples_en.txt")
    path_to_hs_rules_file = Path(path_to_assignment_4, "icamet-samples_hs_rules.txt")

    with open(path_to_hs_file, "r", encoding="utf-8") as hs_file:
        hs_file_contents = hs_file.readlines()[200:]

    with open(path_to_en_file, "r", encoding="utf-8") as en_file:
        en_file_contents = en_file.readlines()[200:]

    with open(path_to_hs_rules_file, "r", encoding="utf-8") as hs_rules_file:
        hs_rules_file_contents = hs_rules_file.readlines()

    print(get_pos_tag_similarity(hs_file_contents, hs_rules_file_contents))
