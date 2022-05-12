import numpy as np
import pandas as pd
import re


file_name = "./data/22330058_ppf_stim_patterns.csv"


def main():
    # file_name = input(f"Enter the file name with full path\n")
    with open(file_name, "r") as f:
        coords = []

        for line in f.readlines():
            line_sub = re.sub("[^0-9,]", "", line)
            coords_list = [int(i) for i in line_sub.split(",")]
            coords.append(np.reshape(coords_list, (int(len(coords_list)/2), 2)))

        np.save(file_name.split(".c")[-2] + ".npy", coords)

            
    

if __name__ == "__main__":
    main()


