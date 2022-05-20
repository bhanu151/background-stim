import numpy as np
import os
from PIL import Image
import protocol_params as prm
import math
import glob
from PPF_analysis import read_data, identify_ppf_patterns


def generate_stim_grid():
    stim_grid = np.ones(prm.GRID_SIZE)
    for i in range(1, prm.GRID_SIZE[0], prm.N_SKIP_ROWS + 1):
        stim_grid[i] = 0
    for i in range(1, prm.GRID_SIZE[0], prm.N_SKIP_COLS + 1):
        stim_grid[:, i] = 0
    # print(stim_grid)
    return stim_grid


def generate_patterns(num_sq):
    stim_grid = generate_stim_grid()
    spots_x, spots_y = np.where(stim_grid == 1)
    # print(spots_x, spots_y)
    n_spots = len(spots_x)
    # print(n_spots)
    patterns = []
    indices = np.arange(n_spots)  # ordered indices
    np.random.shuffle(indices)  # shuffled indices
    # print(indices)
    for i in range(num_sq, n_spots + 1, num_sq):
        # print(i)
        patterns.append(
            np.array(
                [
                    np.array([spots_x[indices[j]], spots_y[indices[j]]])
                    for j in range(i - num_sq, i)
                ]
            )
        )
    # print(patterns)
    # print(len(patterns))
    return np.array(patterns)


def generate_images(patterns, images_path):
    print(patterns[:5])
    if not os.path.isdir(images_path):
        os.mkdir(images_path)
    else:
        if os.listdir(images_path) != []:
            print(
                f"{images_path} already contains some files.\n Please clear them and restart"
            )
            return 0
    for p, pattern in enumerate(patterns):
        image_arr = np.zeros(prm.GRID_SIZE, dtype=np.uint8)
        for coord in pattern:
            x, y = coord
            image_arr[x, y] = np.uint8(255)
        image = Image.fromarray(image_arr)
        image.save(f"{images_path}/image_{p:03}.png")
    return 1


def derange(lst):
    n = len(lst)
    for i in range(n - 1):
        j = np.random.randint(i + 1, n)
        lst[i], lst[j] = lst[j], lst[i]
    return lst


def shuffle_patterns(patterns, num_repeats, batch_size):
    num_batches = math.ceil(patterns.shape[0] / batch_size)
    num_patterns = patterns.shape[0]
    shuffled_patterns = []
    for b in range(num_batches):
        lst = np.arange(b * batch_size, min((b + 1) * batch_size, num_patterns))
        deranged_lst = derange(np.copy(lst))
        # lst = b*batch_size + np.tile(lst, num_repeats)
        # deranged_lst = b*batch_size + np.tile(deranged_lst, num_repeats)
        interspersed_lst = np.zeros(len(lst) + len(deranged_lst), dtype=np.uint16)
        interspersed_lst[::2] = lst
        interspersed_lst[1::2] = deranged_lst
        interspersed_lst = np.tile(interspersed_lst, num_repeats)
        # print(interspersed_lst)
        shuffled_patterns.extend(patterns[interspersed_lst])
    return np.array(shuffled_patterns)


def generate_background_patterns(patterns, num_sq, num_repeats, num_patterns=None):
    # print(patterns)
    if num_patterns:
        spots = patterns.reshape(-1, patterns.shape[-1])[: (num_sq * num_patterns)]
    else:
        spots = patterns.reshape(-1, patterns.shape[-1])
    n_spots = len(spots)
    bg_patterns = []
    for i in range(num_sq, n_spots + 1, num_sq):
        bg_patterns.append(spots[i - num_sq : i])
    bg_patterns = np.tile(np.array(bg_patterns), (num_repeats, 1, 1))
    print(bg_patterns)
    return bg_patterns


def search_for_file(data_path, file_type):
    files = list(filter(os.path.isfile, glob.glob(data_path + "/*" + file_type)))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if len(files) > 0:
        recent_file_flag = input(
            f"{files[0]} is the most recent data file.\nWould you like to analyze this file? y/n\n"
        )
        if recent_file_flag == "y":
            data_file = files[0]
        else:
            while True:
                data_file = (
                    data_path + "/" + input(f"Enter the {file_type} data file name\n")
                )
                if not os.path.isfile(data_file):
                    print("File not found. Try again.")
                    continue
                break
    else:
        while True:
            data_file = (
                data_path + "/" + input(f"Enter the {file_type} data file name\n")
            )
            if not os.path.isfile(data_file):
                print("File not found. Try again.")
                continue
            break
    return data_file


def run_ppf_batch(data_path, experiment_name, patterns_batch, batch_num):
    ppf_proto_complete_flag = "n"
    while ppf_proto_complete_flag != "y" and ppf_proto_complete_flag != "Y":
        ppf_proto_complete_flag = input("Has PPF run completed? y/n\n")

    np.save(
        f"{data_path}/{experiment_name}_ppf_stim_patterns_batch_{batch_num:02}.npy",
        patterns_batch,
    )

    abf_data_file = search_for_file(data_path, ".abf")
    print()
    patterns_data_file = search_for_file(data_path, ".npy")
    time_points, raw_data, patterns = read_data(abf_data_file, patterns_data_file)
    ppf, no_ppf = identify_ppf_patterns(time_points, raw_data, patterns)
    return ppf, no_ppf


def main():
    data_path = input("Enter the path to the data\n")
    experiment_name = input("Enter the experiment name\n")
    print("STEP 1 : PPF PATTERN GENERATION")
    print("===============================")
    num_sq = int(
        input("How many squares required in each pattern for PPF stim protocol?\n")
    )
    patterns = generate_patterns(num_sq)
    num_repeats = int(input("How many repeats of each pattern?\n"))
    batch_size = int(input("How many patterns you want to simulate in a batch?\n"))
    shuffled_patterns = shuffle_patterns(patterns, num_repeats, batch_size)
    np.save(
        data_path + "/" + experiment_name + "_ppf_stim_patterns_full_set.npy",
        shuffled_patterns,
    )
    images_path = data_path + "/ppf_stim_patterns"
    images_flag = generate_images(shuffled_patterns, images_path)
    if not images_flag:
        return 0

    print(f"Generated a total of {len(shuffled_patterns)} images")
    print(
        "*******************************************************************************************"
    )
    print()

    print("STEP 2 : PPF STIMULATION")
    print("========================")
    completed_images_count = 0
    ppf_patterns = []
    no_ppf_patterns = []
    while completed_images_count < len(shuffled_patterns):

        run_ppf_proto_flag = input(
            f"Would you like to run the PPF stim protocol for the next {batch_size} patterns? y/n\n"
        )
        batch_num = 0
        if run_ppf_proto_flag == "y" or run_ppf_proto_flag == "Y":

            patterns_batch = shuffled_patterns[
                batch_num
                * batch_size
                * num_repeats
                * 2 : (batch_num + 1)
                * batch_size
                * num_repeats
                * 2
            ]
            ppf, no_ppf = run_ppf_batch(
                data_path, experiment_name, patterns_batch, batch_num
            )
            ppf_patterns.extend(ppf)
            no_ppf_patterns.extend(no_ppf)
            batch_num += 1
            completed_images_count += len(patterns_batch)
            print(
                f"#completed patterns = {int(completed_images_count / (num_repeats*2))}\n"
            )
            print(f"#PPF patterns = {len(ppf_patterns)}\n")
            print(f"#No PPF patterns = {len(no_ppf_patterns)}\n")

        elif run_ppf_proto_flag == "n" or run_ppf_proto_flag == "N":
            break
    print(
        "*******************************************************************************************"
    )
    print()
    bg_proto_code = input(
        "Would you like to run background stimulation for\n\
        1. PPF patterns\n 2. no PPF patterns\n 3. Wrap up experiment\n\
        Enter option 1 or 2 or 3\n"
    )
    while bg_proto_code != "1" and bg_proto_code != "2" and bg_proto_code != "3":
        print("Invalid input. Please retry!")
        bg_proto_code = input(
            "Would you like to run background stimulation for\n\
            1. PPF patterns\n 2. no PPF patterns\n 3. Wrap up experiment\n\
            Enter option 1 or 2 or 3\n"
        )
    if bg_proto_code == "1" or bg_proto_code == "2":
        print("STEP 3 : BACKGROUND PATTERN GENERATION")
        print("======================================")
        num_sq_bg = int(
            input(
                "How many squares required in each pattern for background stim protocol?\n"
            )
        )
        num_bg_patterns = int(input("How many background patterns are required?\n"))
        num_bg_repeats = int(input("How many repeats of each pattern?\n"))
        if bg_proto_code == "1":
            bg_patterns = generate_background_patterns(
                np.array(ppf_patterns),
                num_sq=num_sq_bg,
                num_repeats=num_bg_repeats,
                num_patterns=num_bg_patterns,
            )
            bg_images_path = data_path + "/bg_stim_patterns_ppf_spots"
        if bg_proto_code == "2":
            bg_patterns = generate_background_patterns(
                np.array(no_ppf_patterns),
                num_sq=num_sq_bg,
                num_repeats=num_bg_repeats,
                num_patterns=num_bg_patterns,
            )
            bg_images_path = data_path + "/bg_stim_patterns_no_ppf_spots"
        bg_images_flag = generate_images(bg_patterns, bg_images_path)
        if not bg_images_flag:
            return 0

        print(f"Generated a total of {len(bg_patterns)} images")

    print(
        "*******************************************************************************************"
    )
    print("Experiment Complete")


if __name__ == "__main__":
    main()
