"""
Extracts the seed permutations from the log files in the logs folder and saves
 them to a pickle file.

Dan Forbes and Lara Stroh - 2024
"""


def main():
    import pickle
    from pathlib import Path

    logs_folder = Path("logs")
    permutations = {}
    num_files = 0

    for problem_dir in logs_folder.iterdir():
        for cluster_dir in problem_dir.iterdir():
            print(cluster_dir)
            if not cluster_dir.is_dir():
                continue

            log_files = list(cluster_dir.glob('*.txt'))
            if not log_files:
                continue

            latest_log_file = max(log_files, key=lambda f: f.stat().st_ctime)
            print(f"  {latest_log_file}")
            with latest_log_file.open() as file:
                lines = file.readlines()

            num_files += 1
            for i, line in enumerate(lines):
                if "The final seed permutation is" not in line:
                    continue

                # Convert the next line to a list of ints
                # E.g. "[1, 2, 3, 4, 5]" -> [1, 2, 3, 4, 5]
                # This is a security risk, but it's fine for our purposes...
                permutation = eval(lines[i + 1])

                key = (problem_dir.stem, int(cluster_dir.stem.split("_")[0]))
                permutations[key] = permutation
                break
            else:
                key = (problem_dir.stem, int(cluster_dir.stem.split("_")[0]))
                permutations[key] = None

    # Pickle permutations
    save_filename = Path("permutations.pickle")

    if len(permutations) != num_files:
        raise Exception(
            f"Missing Permutations, {len(permutations)} != {num_files}")

    with open(save_filename, "wb") as file:
        pickle.dump(permutations, file)

    print(f"Saved permutations to {save_filename}")


if __name__ == "__main__":
    main()
