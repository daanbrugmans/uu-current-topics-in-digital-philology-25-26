import string
from pathlib import Path


def get_exact_match_accuracy(hypotheses: list[str], references: list[str]) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses and the references should be the same.")

    exact_match_count = 0
    total_match_count = 0
    for hypothesis, reference in zip(hypotheses, references):
        punctuation_chars = list(string.punctuation)
        punctuation_chars.extend([" ", "\n"])

        is_all_punctuation = True
        hypothesis_chars = hypothesis.split(" ")
        for char in hypothesis_chars:
            if char not in punctuation_chars:
                is_all_punctuation = False
                break

        if is_all_punctuation:
            continue

        preprocessed_hypothesis = hypothesis.replace(" ", "").replace("\n", "")
        preprocessed_reference = reference.replace(" ", "").replace("\n", "")

        if preprocessed_hypothesis == preprocessed_reference:
            exact_match_count += 1

        total_match_count += 1

    return round(exact_match_count / total_match_count * 100, 1)


if __name__ == "__main__":
    path_to_assignment_5 = Path(__file__).parents[0]
    path_to_swedish_data = Path(path_to_assignment_5, "data", "swedish")

    with open(
        Path(path_to_swedish_data, "swedish.hs-sv.test.tok.sv"), "r", encoding="utf-8"
    ) as modern_swedish_test_file:
        modern_swedish_test_text = modern_swedish_test_file.readlines()

    with open(
        Path(path_to_swedish_data, "swedish.hs-sv.test.tok.hs"), "r", encoding="utf-8"
    ) as historical_swedish_test_file:
        historical_swedish_test_text = historical_swedish_test_file.readlines()

    with open(
        Path(path_to_swedish_data, "out.sv"), "r", encoding="utf-8"
    ) as translated_swedish_test_file:
        translated_swedish_test_text = translated_swedish_test_file.readlines()

    print(
        get_exact_match_accuracy(historical_swedish_test_text, modern_swedish_test_text)
    )
    print(
        get_exact_match_accuracy(translated_swedish_test_text, modern_swedish_test_text)
    )
