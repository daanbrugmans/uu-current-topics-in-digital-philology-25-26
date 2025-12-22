from pathlib import Path

if __name__ == "__main__":
    path_to_assignment_5 = Path(__file__).parents[0]
    path_to_swedish_data = Path(path_to_assignment_5, "data", "swedish")

    set_names = ["train", "test", "dev"]
    version_names = ["hs", "sv"]

    for set_name in set_names:
        for version_name in version_names:
            filename = f"swedish.hs-sv.{set_name}.{version_name}"
            with open(
                Path(path_to_swedish_data, filename), "r", encoding="utf-8"
            ) as swedish_data_file:
                swedish_data = swedish_data_file.readlines()

            preprocessed_swedish_data = []
            for line in swedish_data:
                if not line:
                    continue

                if line == "\n":
                    continue

                line_with_spaces = " ".join(list(line))
                preprocessed_swedish_data.append(line_with_spaces)

            preprocessed_filename = f"swedish.hs-sv.{set_name}.tok.{version_name}"
            with open(
                Path(path_to_swedish_data, preprocessed_filename), "w", encoding="utf-8"
            ) as preprocessed_swedish_data_file:
                if set_name == "test":
                    preprocessed_swedish_data = preprocessed_swedish_data[:2999]

                preprocessed_swedish_data_file.write("".join(preprocessed_swedish_data))
