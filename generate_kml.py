import pandas as pd

def geohash2kml(df):
    # get geo info
    geo_info = df.geohash6.value_counts()

    # save info to file
    geo_info.to_csv("geo_info.txt", header=None, sep=" ")

    import os
    os.system('python geohash2kml.py geo_info.txt singapore_geo.kml')

if __name__ == "__main__":

    # read file
    df = pd.read_csv("data/training.csv")

    # generate kml
    geohash2kml(df)
