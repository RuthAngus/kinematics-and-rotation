#!/usr/bin/python3

import numpy as np
import pandas as pd
import stardate as sd
import sys
import os

# Add cwd to path for SLURM because it executes a copy
# sys.path.append(os.getcwd())

from multiprocessing import Pool

def infer_stellar_age(df):

    if df["n_[Fe/H]i"] == "SPE":
        iso_params = {"g": (df["g_final"], df["g_final_err"]),
                      "r": (df["r_final"], df["r_final_err"]),
                      "J": (df["Jmag"], df["e_Jmag"]),
                      "H": (df["Hmag"], df["e_Hmag"]),
                      "K": (df["Kmag"], df["e_Kmag"]),
                      "feh": (df["[Fe/H]"], .05),
                      "parallax": (df["plx"]*1e3, df["plxe"]*1e3)}
    else:
        iso_params = {"g": (df["g_final"], df["g_final_err"]),
                      "r": (df["r_final"], df["r_final_err"]),
                      "J": (df["Jmag"], df["e_Jmag"]),
                      "H": (df["Hmag"], df["e_Hmag"]),
                      "K": (df["Kmag"], df["e_Kmag"]),
                      "parallax": (df["plx"]*1e3, df["plxe"]*1e3)}

    prot, prot_err = df["Prot"], df["Prot"]*.05
    star = sd.Star(iso_params, prot=prot, prot_err=prot_err,
                   Av=df["Av"], Av_err=df["Av_std"],
                   savedir="mcquillan", filename="{}".format(df["KID"]))

    star.fit(max_n=1000, thin_by=100)


if __name__ == "__main__":
    df = pd.read_csv("single_MS_stars.csv")
    df = df.iloc[:4]

    list_of_dicts =  []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())
    print(list_of_dicts[0])
    print(len(list_of_dicts))

    p = Pool(4)
    list(p.map(infer_stellar_age, list_of_dicts))
