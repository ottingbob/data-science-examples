# TODO: Get the actual CSV file from netflix...
# https://medium.com/python-point/python-analyze-your-own-netflix-data-29bcc351bb08
import datetime
import random
from typing import Any, List

import faker
import matplotlib.pyplot as plt
import polars as pl

columns = [
    "Profile Name",
    "Start Time",
    "Duration",
    "Attributes",
    "Title",
    "Supplemental Video Type",
    "Device Type",
    "Bookmark",
    "Latest Bookmark",
    "Country",
]

random.seed(5252)
faker.Faker.seed(5252)
fake = faker.Faker()

num_records = 9273

profile_names = [fake.name() for _ in range(5)]
start_time: datetime.datetime = fake.date_time_between(start_date="-5y")
duration = fake.time_delta()
known_attrs = [None, "Autoplayed: user", "action: None", "action: User_Interaction"]
titles = [fake.sentence() for _ in range(int(num_records / 3))]
video_types = [None, "HOOK"]
device_type = [
    "Samsung CE 2020 Nike-L UHD TV Smart TV",
    "Internet Explorer (Cadmium)",
    "PC",
    "Sony CE Sony Android TV 2020 M5 Smart TV",
    "FireTV 4K Stick 2018",
    "Chrome PC (Cadmium)",
    "Sony 2012 Blu-ray Players",
    "DefaultWidevineAndroidTablets",
    "Apple iPad Pro 12.9 in 5th Gen (Wi-Fi/Cell) iPad",
]
bookmark = fake.time_delta(end_datetime=(datetime.timedelta(hours=4)))
latest_bookmark = [
    fake.time_delta(end_datetime=(datetime.timedelta(hours=4))),
    "Not latest view",
]
country = fake.country()


def make_fake_row() -> List[Any]:
    return [
        profile_names[random.randint(0, len(profile_names) - 1)],
        fake.date_time_between(start_date="-5y"),
        fake.time_delta(end_datetime=(datetime.timedelta(hours=8))),
        known_attrs[random.randint(0, len(known_attrs) - 1)],
        titles[random.randint(0, len(titles) - 1)],
        video_types[random.randint(0, 1)],
        device_type[random.randint(0, len(device_type) - 1)],
        fake.time_delta(end_datetime=(datetime.timedelta(hours=4))),
        fake.time_delta(end_datetime=(datetime.timedelta(hours=4)))
        if random.randint(0, 1) == 0
        else "Not latest view",
        fake.country(),
    ]


rows = []
for _ in range(num_records):
    rows.append(make_fake_row())

# Article has a DF of 9273 rows so we can fake that many...
netflix_data = {}
for idx, col_name in enumerate(columns):
    netflix_data[col_name] = [rows[row_idx][idx] for row_idx in range(len(rows))]

pl.Config.set_tbl_cols(len(columns))
netflix_df = pl.DataFrame(netflix_data)
print(netflix_df)

# Look at the unique profile names:
print(netflix_df.select("Profile Name").unique())
# Unique device types
print(netflix_df.select("Device Type").unique())

# Here are some questions to answer:
# - Which profile watched the most time
# - Which profile has the most watching activities / interactions
# - What is the average watching time (per profile)
# - What devices are used by which profile & which device is used the most


# - Which profile watched the most time
def get_profile_watch_time():
    watch_time = (
        netflix_df.groupby(pl.col("Profile Name"))
        .agg([pl.col("Duration").sum().alias("Total Duration")])
        .sort("Total Duration", descending=True)
    )
    print(watch_time)
    plt.title("Profile Total Watch Duration")
    plt.bar(watch_time["Profile Name"], watch_time["Total Duration"])
    plt.show()


# - Which profile has the most watching activities / interactions
def get_profile_interactions():
    interactions = (
        netflix_df.groupby(pl.col("Profile Name"))
        .agg([pl.col("Profile Name").count().alias("Total Interactions")])
        .sort("Total Interactions", descending=True)
    )
    print(interactions)
    plt.title("Profile Total Interactions")
    plt.bar(interactions["Profile Name"], interactions["Total Interactions"])
    plt.show()


get_profile_watch_time()
get_profile_interactions()

# - What is the average watching time (per profile)
print(
    netflix_df.groupby(pl.col("Profile Name"))
    .agg(
        [
            pl.col("Duration").sum().alias("Total Duration"),
            pl.col("Profile Name").count().alias("Total Interactions"),
        ]
    )
    .with_columns(
        pl.struct(["Total Duration", "Total Interactions"])
        .apply(
            lambda x: datetime.timedelta(
                seconds=round(
                    x["Total Duration"].total_seconds() / x["Total Interactions"], 2
                )
            )
        )
        .alias("Average Watching Time")
    )
    .select(["Profile Name", "Average Watching Time"])
    .sort("Average Watching Time", descending=True)
)

# - What devices are used by which profile & which device is used the most
print(
    netflix_df.groupby([pl.col("Profile Name"), pl.col("Device Type")])
    .agg([pl.col("Device Type").count().alias("Device Count")])
    .filter(pl.col("Device Count") == pl.col("Device Count").max().over("Profile Name"))
    .sort(["Device Count", "Profile Name"], descending=True)
)

# TODO: Here are some more questions:
# - What was the most popular/watched title?
# - Was there any title watched by all Profiles?
# - Can we recommend a title for one Profile based on the common watching history of other Profiles?
