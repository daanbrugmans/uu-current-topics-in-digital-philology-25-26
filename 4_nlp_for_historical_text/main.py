import re
from pathlib import Path

import nltk


def print_nltk_pos_tags():
    with open(path_to_hs_file, "r", encoding="utf-8") as hs_file:
        hs_file_contents = hs_file.readlines()

        for i, sentence in enumerate(hs_file_contents):
            print(i)
            print("  ", nltk.tag.pos_tag(nltk.tokenize.word_tokenize(sentence)))


if __name__ == "__main__":
    path_to_assignment_4 = Path(__file__).parents[0]
    path_to_hs_file = Path(path_to_assignment_4, "icamet-samples_hs.txt")
    path_to_en_file = Path(path_to_assignment_4, "icamet-samples_en.txt")

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
