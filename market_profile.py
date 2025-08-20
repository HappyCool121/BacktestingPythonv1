# this file is an exploration of market profile, as well as market profile concepts
from typing import Callable

from datahandler import DataHandler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# get data
symbols = ["ES=F"]
symbol_to_test = "ES=F"
CONFIG = {
    "symbols": symbols,  # Pass the symbol as a list
    "start_date": "2025-08-15",  # Using a single day for this example
    "end_date": "2025-08-16",
    "interval": "30m",
    "market_profile_type": 6,
    "iteration_choice": 2
}

pd.set_option('display.max_columns', None)

# 1. PREPARE DATA AND SIGNALS
# DataHandler fetches data and returns a dictionary
data_handler = DataHandler(
    symbols=CONFIG["symbols"],
    start_date=CONFIG["start_date"],
    end_date=CONFIG["end_date"],
    interval=CONFIG["interval"]
)
price_data_dict = data_handler.get_data()
df = price_data_dict[symbol_to_test]

# --- Workflow Expansion ---
# Data Preparation and TPO Assignment
# 1. create dataframe for TPO coordinates: -----------------------------------
tpo_data_columns = ['datetime', 'price', 'tpo']
tpo_df = pd.DataFrame(columns=tpo_data_columns)
print(tpo_df)

# 2. Separate into regular trading hours and overnight trading hours :-------------------------
start_time = CONFIG["start_date"] + ' 09:30+00:00'
end_time = CONFIG["start_date"] + ' 16:00+00:00'
# print(start_time, end_time)

truncated_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
truncated_df = truncated_df.reset_index(drop=True)
outside_hours_df = df[(df['date'] < start_time) | (df['date'] > end_time)]
outside_hours_df = outside_hours_df.reset_index(drop=True)

print("TRUNCATED DATA, trading hours")
print(truncated_df)
print("OUTSIDE HOURS")
print(outside_hours_df)

# overall flow of the code:
# 1. get the data, full day intraday data, interval of 30 mins
# 2. separate the data either into regular trading hours or overnight trading hours
# 3. assign letters to the truncated time price data
# 4. create the point dataframe, either for the consolidated or disintegrated market profile
# 5. plot the points
# 5.5. plot value area high value area low

"""
workflow on creating the market profile:

first things first we will draw bar like charts, but with the tpo letters;
so the x axis would be hte datetime values, and the y axis would be the letters,
a single period will include letters populated in between the high and low of that period

1. iterate through the dataframe 
2. at an iteration of one period, we will determine the high and low, as well as the letter
3. then we would iterate through the range according to a given tick, and create points at which
    the letter representing that period will fill
4. for that period, we have a list of coordinates for that particular letter to fill
5. repeat for all periods
6. plot the letters according to these coordinates

*as of right now, we will be ignoring the POC, value area, as well as opening balance


so basically what i need is a dataframe containing: price, time (datetime, but more specifically time)
and the letter corresponding to the period

so to summarize the problem: 

given a time price data, which include OHLC values with an interval of 30m, create a market profile diagram 
each period (interval of 30 minutes) is assigned a letter, ranging from ABCD.......xyz, and for each period
the letters assigned to that period will populate that period from the low to the high of that period. 

one way we can approach this problem is to create a simple dataframe containing every single point of the market 
profile, which means each letter has to be accounted for, and the data will be represented in the form of a dataframe
with 3 different columns; datetime, price point, and letter. this would be a comically large dataframe, but it 
would work anyway (i think)

how would we go about doing this?

first iterate through the dataframe, 
for each period, find the high - low, and divide by the tick size to determine how many points for this specific range
then we iterate through the range, and add it to the main coordinate dataframe.
repeat for all periods

 

"""
"""
# trying to plot:

# --- Data Preparation ---
data = {
    'x_val': [1, 2, 3, 4, 5, 6, 7, 8],
    'y_val': [5, 4, 8, 6, 9, 7, 2, 3],
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
}
df = pd.DataFrame(data)

print('trying to plot this')
print(df)

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(8, 6))

# Define a dictionary to map categories to markers 📌
# 'o' is a circle, '^' is a triangle up, 's' is a square, etc.
markers = {'A': 'o', 'B': '^'}
colors = {'A': 'blue', 'B': 'green'}

# --- Loop and Plot ---
# Group by 'category' and loop through each group
for category, group_df in df.groupby('category'):
    ax.scatter(
        group_df['x_val'],
        group_df['y_val'],
        marker=markers[category],
        color=colors[category],
        label=f'Category {category}',
        s=100  # s is the marker size
    )

# --- Customization ---
ax.set_title("Scatter Plot with Different Markers per Category")
ax.set_xlabel("X Value")
ax.set_ylabel("Y Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# --- Show the Plot ---
plt.show()

"""

def calculate_market_profile_levels(tpo_df: pd.DataFrame):

    # Step 1: Get TPO counts for each price level
    tpo_counts = tpo_df['price'].value_counts().sort_index()

    # Step 2: Find the Point of Control (POC)
    poc = tpo_counts.idxmax()

    # Step 3: Calculate the Value Area (70% of TPOs)
    total_tpos = len(tpo_df)
    value_area_tpos_target = total_tpos * 0.7

    # Start building the Value Area from the POC
    value_area_prices = [poc]
    current_value_area_tpos = tpo_counts[poc]

    # Get price levels above and below the POC, sorted by proximity
    prices_below = sorted([p for p in tpo_counts.index if p < poc], reverse=True)
    prices_above = sorted([p for p in tpo_counts.index if p > poc])

    # Iteratively expand the Value Area until 70% of TPOs are included
    while current_value_area_tpos < value_area_tpos_target:
        # Find the next price level to add (choose the one with more TPOs)
        count_below = tpo_counts.get(prices_below[0], 0) if prices_below else 0
        count_above = tpo_counts.get(prices_above[0], 0) if prices_above else 0

        if count_below > count_above:
            price_to_add = prices_below.pop(0)
            current_value_area_tpos += count_below
        else:
            price_to_add = prices_above.pop(0)
            current_value_area_tpos += count_above

        value_area_prices.append(price_to_add)

        # Break if we run out of prices to check
        if not prices_below and not prices_above:
            break

    vah = max(value_area_prices)
    val = min(value_area_prices)

    dict = {
        'poc': poc,
        'vah': vah,
        'val': val
    }
    return dict

def create_market_profile_coordinates(df: pd.DataFrame):

    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    df['tpo'] = [tpo_letters[i] for i in range(len(df))]

    # --- 3. Generate Data for Disintegrated Profile ---
    disintegrated_rows = []
    for index, row in df.iterrows():
        high, low, tpo_letter = row['high'], row['low'], row['tpo']
        tick_size = 0.25
        price_range = int((high - low) / tick_size)
        low = math.ceil(low)
        for number in range(price_range):
            if number % 4 == 0 and low < high:
                price = low + (number * tick_size)
                disintegrated_rows.append({'datetime': index, 'price': price, 'tpo': tpo_letter})
    disintegrated_tpo_df = pd.DataFrame(disintegrated_rows)

    # --- 4. Generate Data for Consolidated Profile ---
    consolidated_rows = []
    price_level_occupancy = {}
    for index, row in df.iterrows():
        high, low, tpo_letter = row['high'], row['low'], row['tpo']
        tick_size = 1.0
        low = math.ceil(low)
        price_points = np.arange(low, high, tick_size)
        for price in price_points:
            price = round(price / tick_size) * tick_size
            x_pos = price_level_occupancy.get(price, 0)
            consolidated_rows.append({'datetime': x_pos, 'price': price, 'tpo': tpo_letter})
            price_level_occupancy[price] = x_pos + 1
    consolidated_tpo_df = pd.DataFrame(consolidated_rows)

    dict = {
        'consolidated_tpo_df': consolidated_tpo_df,
        'disintegrated_tpo_df': disintegrated_tpo_df
    }

    return dict

def get_key_values(consolidated_tpo_df: pd.DataFrame, disintegrated_tpo_df: pd.DataFrame, calculate_market_profile_levels):
    # --- 5. Get key levels before plotting ---
    results = calculate_market_profile_levels(consolidated_tpo_df)
    poc = results['poc']
    vah = results['vah']
    val = results['val']

def plot_coordinates (disintegrated_tpo_df: pd.DataFrame, consolidated_tpo_df: pd.DataFrame, type: int, colourVAL: str = 'green'):
    # --- 6. Create Side-by-Side Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 10), sharey=True)

    # Plot 1: Disintegrated Profile
    for index, row in disintegrated_tpo_df.iterrows():
        ax1.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax1.set_title("Disintegrated TPO Chart")
    ax1.set_xlabel("Time Period Index")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, disintegrated_tpo_df['datetime'].max() + 1)
    ax1.set_ylim(df['low'].min() - 1, df['high'].max() + 1)
    ax1.axhspan(val, vah, color='gray', alpha=0.2)
    ax1.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax1.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax1.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax1.legend()

    # Plot 2: Consolidated Profile
    for index, row in consolidated_tpo_df.iterrows():
        color = 'green' if val <= row['price'] <= vah else 'blue'
        ax2.scatter(x=row['datetime'], y=row['price'], marker='s', s=100, c=color)
    ax2.set_title("Consolidated Market Profile")
    ax2.set_xlabel("TPO Count")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(-1, consolidated_tpo_df['datetime'].max() + 1)
    ax2.axhspan(val, vah, color='gray', alpha=0.2, label='Value Area (70%)')
    ax2.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax2.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax2.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_market_profile(ohlc_df: pd.DataFrame, trading_date: str, calculate_market_profile_levels: bool = True):
    """
    Generates and displays a side-by-side Disintegrated and Consolidated Market Profile chart.

    Args:
        ohlc_df (pd.DataFrame): The input DataFrame with OHLC data and a 'date' column.
        trading_date (str): The date to analyze in 'YYYY-MM-DD' format.
        calculate_market_profile_levels (function): The helper function to calculate POC, VAH, and VAL.
    """
    # --- 1. Truncate DataFrame to the specified trading day ---
    start_time = trading_date + ' 09:30+00:00'
    end_time = trading_date + ' 16:00+00:00'

    truncated_df = ohlc_df[(ohlc_df['date'] >= start_time) & (ohlc_df['date'] <= end_time)]
    truncated_df = truncated_df.reset_index(drop=True)

    # create overnight trading hours data:
    outside_hours_df = df[(df['date'] < start_time) | (df['date'] > end_time)]

    print("TRUNCATED DATA, trading hours")
    print(truncated_df)

    print("OUTSIDE HOURS")
    print(outside_hours_df)

    # --- 2. Assign TPO letters ---
    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    truncated_df['tpo'] = [tpo_letters[i] for i in range(len(truncated_df))]

    # --- 3. Generate Data for Disintegrated Profile ---
    disintegrated_rows = []
    for index, row in truncated_df.iterrows():
        high, low, tpo_letter = row['high'], row['low'], row['tpo']
        tick_size = 0.25
        price_range = int((high - low) / tick_size)
        low = math.ceil(low)
        for number in range(price_range):
            if number % 4 == 0 and low < high:
                price = low + (number * tick_size)
                disintegrated_rows.append({'datetime': index, 'price': price, 'tpo': tpo_letter})
    disintegrated_tpo_df = pd.DataFrame(disintegrated_rows)

    # --- 4. Generate Data for Consolidated Profile ---
    consolidated_rows = []
    price_level_occupancy = {}
    for index, row in truncated_df.iterrows():
        high, low, tpo_letter = row['high'], row['low'], row['tpo']
        tick_size = 1.0
        low = math.ceil(low)
        price_points = np.arange(low, high, tick_size)
        for price in price_points:
            price = round(price / tick_size) * tick_size
            x_pos = price_level_occupancy.get(price, 0)
            consolidated_rows.append({'datetime': x_pos, 'price': price, 'tpo': tpo_letter})
            price_level_occupancy[price] = x_pos + 1
    consolidated_tpo_df = pd.DataFrame(consolidated_rows)

    # --- 5. Get key levels before plotting ---
    results = calculate_market_profile_levels(consolidated_tpo_df)
    poc = results['poc']
    vah = results['vah']
    val = results['val']

    # --- 6. Create Side-by-Side Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 10), sharey=True)

    # Plot 1: Disintegrated Profile
    for index, row in disintegrated_tpo_df.iterrows():
        ax1.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax1.set_title("Disintegrated TPO Chart")
    ax1.set_xlabel("Time Period Index")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, disintegrated_tpo_df['datetime'].max() + 1)
    ax1.set_ylim(truncated_df['low'].min() - 1, truncated_df['high'].max() + 1)
    ax1.axhspan(val, vah, color='gray', alpha=0.2)
    ax1.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax1.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax1.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax1.legend()

    # Plot 2: Consolidated Profile
    for index, row in consolidated_tpo_df.iterrows():
        color = 'green' if val <= row['price'] <= vah else 'blue'
        ax2.scatter(x=row['datetime'], y=row['price'], marker='s', s=100, c=color)
    ax2.set_title("Consolidated Market Profile")
    ax2.set_xlabel("TPO Count")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(-1, consolidated_tpo_df['datetime'].max() + 1)
    ax2.axhspan(val, vah, color='gray', alpha=0.2, label='Value Area (70%)')
    ax2.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax2.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax2.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# -------------------- Calculation and sketching of profiles --------------------

# disintegrated profile
if CONFIG["market_profile_type"] == 0:

    truncated_df['tpo'] = ''
    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for i in range(len(truncated_df)):
        truncated_df.loc[truncated_df.index[i], 'tpo'] = tpo_letters[i]

    print('df after assigning TPOs')
    print(truncated_df[['date', 'tpo']])

    counter = 0
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']

        price_range = int((row['high'] - row['low'])/0.25) #maximum number of price points (per tick)
        # print(f'no of points for {symbol_to_test} at {index}: {price_range}')
        # print(f'current count: {counter}')

        iteration_choice = CONFIG['iteration_choice'] #plot every point by default
        for number in range(price_range):
            # iterate through the number of price ticks

            print(number) # just to check
            # we will be doing 3 different ways to plot: every tick, every 2 ticks, every 4 ticks (one point)
            if iteration_choice == 0: # every tick:
                # can just start with the price as it is
                print(f'Iterating every tick, so low = {low}')

                tpo_df.loc[counter, 'price'] = low + (number*0.25)
                tpo_df.loc[counter, 'tpo'] = row['tpo'] # current rows' letter
                tpo_df.loc[counter, 'datetime'] = index # using INDEX instead of DATETIME !!!
                counter += 1

            elif iteration_choice == 1: # every 2 ticks
                # will have to start with the first .0 or .5
                low = round(low * 2) / 2 # nearest .5
                print(f'iterating every 2 ticks, so low = {low}')

                if number % 2==0 and low < high:
                    tpo_df.loc[counter, 'price'] = low + (number * 0.25)
                    tpo_df.loc[counter, 'tpo'] = row['tpo']  # current rows' letter
                    tpo_df.loc[counter, 'datetime'] = index  # using INDEX instead of DATETIME !!!
                    counter += 1

            elif iteration_choice == 2: # every point:
                # will have to start with the nearest larger whole number
                low = math.ceil(low) # nearest whole number
                print(f'iteerating every point, so low = {low}')

                if number % 4==0 and low < high:
                    tpo_df.loc[counter, 'price'] = low + (number * 0.25)
                    tpo_df.loc[counter, 'tpo'] = row['tpo']  # current rows' letter
                    tpo_df.loc[counter, 'datetime'] = index  # using INDEX instead of DATETIME !!!
                    counter += 1

    # print(tpo_df.head)
    # print(tpo_df.tail)

    fig, ax = plt.subplots(figsize=(10, 14))

    datetimecounter = 0
    # Loop through the DataFrame to plot each letter
    for index, row in tpo_df.iterrows():
        datetimecounter = row['datetime']
        ax.text(
            x=row['datetime'],
            y=row['price'],
            s=row['tpo'],
            ha='center',
            va='center',
            fontsize=9
        )

    ax.set_title("TPO Chart")
    ax.set_xlabel("Time Price Opportunity (datetime)")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Set the x-axis limits to go from 0 to 45 with a little padding
    ax.set_xlim(-1, datetimecounter + 1)

    # Adjust y-axis limits for better visibility
    ax.set_ylim(tpo_df['price'].min() - 2, tpo_df['price'].max() + 2)
    # plt.savefig('disintegratedmarketprofile.png', dpi=300, bbox_inches='tight')
    plt.show()

# consolidated market profile
elif CONFIG["market_profile_type"] == 1:

    truncated_df['tpo'] = ''
    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for i in range(len(truncated_df)):
        truncated_df.loc[truncated_df.index[i], 'tpo'] = tpo_letters[i]

    print('df after assigning TPOs')
    print(truncated_df[['date', 'tpo']])

    # dict which has price levels as keys and number of points at that price level for teh value
    price_level_occupancy = {}

    for index, row in truncated_df.iterrows(): #iterate every index (every period)

        high = row['high']
        low = row['low']
        tick_size = 1.0

        low = math.ceil(low)  # nearest whole number
        print(f'iterating every point, so low = {low}')

        # we will be using price per point for the first example
        price_points = (np.arange(low, high, tick_size))
        print(f'price pointes output: {price_points}')

        for price in price_points: # iterate through price range for the period
            price = round(price / tick_size) * tick_size
            print(f'current price: {price}')

            # Get the next available horizontal slot (x-coordinate) for this price
            # If the price is new, it starts at slot 0. Otherwise, increment the count.
            x_pos = price_level_occupancy.get(price, 1)

            # Add the new point to our plotting DataFrame
            new_row = {'datetime': x_pos, 'price': price, 'tpo': row['tpo']}
            tpo_df = pd.concat([tpo_df, pd.DataFrame([new_row])], ignore_index=True)

            # Update the occupancy count for this price level
            price_level_occupancy[price] = x_pos + 1

    # print(tpo_df.head)
    # print(tpo_df.tail)

    fig, ax = plt.subplots(figsize=(10, 14))

    datetimecounter = 0
    # Loop through the DataFrame to plot each letter
    for index, row in tpo_df.iterrows():
        datetimecounter = row['datetime']
        ax.text(
            x=row['datetime'],
            y=row['price'],
            s=row['tpo'],
            ha='center',
            va='center',
            fontsize=9
        )

    ax.set_title("TPO Chart")
    ax.set_xlabel("Time Price Opportunity (datetime)")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Set the x-axis limits to go from 0 to 45 with a little padding
    ax.set_xlim(-1, datetimecounter + 10)

    # Adjust y-axis limits for better visibility
    ax.set_ylim(tpo_df['price'].min() - 2, tpo_df['price'].max() + 2)
    # plt.savefig('disintegratedmarketprofile.png', dpi=300, bbox_inches='tight')
    plt.show()

# both profiles
elif CONFIG["market_profile_type"] == 2:

    truncated_df['tpo'] = ''
    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for i in range(len(truncated_df)):
        truncated_df.loc[truncated_df.index[i], 'tpo'] = tpo_letters[i]

    print('df after assigning TPOs')
    print(truncated_df[['date', 'tpo']])

    # --- Generate Data for Disintegrated Profile (Plot 1) ---
    disintegrated_tpo_df = pd.DataFrame(columns=['datetime', 'price', 'tpo'])
    counter = 0
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']
        tick_size = 0.25
        price_range = int((high - low) / tick_size)
        low = math.ceil(low)
        for number in range(price_range):
            if number % 4 == 0 and low < high:
                new_row = {'datetime': index, 'price': low + (number * tick_size), 'tpo': row['tpo']}
                disintegrated_tpo_df = pd.concat([disintegrated_tpo_df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Generate Data for Consolidated Profile (Plot 2) ---
    consolidated_tpo_df = pd.DataFrame(columns=['datetime', 'price', 'tpo'])
    price_level_occupancy = {}
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']
        tick_size = 1.0
        low = math.ceil(low)
        price_points = np.arange(low, high, tick_size)
        for price in price_points:
            price = round(price / tick_size) * tick_size
            x_pos = price_level_occupancy.get(price, 0)
            new_row = {'datetime': x_pos, 'price': price, 'tpo': row['tpo']}
            consolidated_tpo_df = pd.concat([consolidated_tpo_df, pd.DataFrame([new_row])], ignore_index=True)
            price_level_occupancy[price] = x_pos + 1

    # get key levels before plotting
    result = calculate_market_profile_levels(consolidated_tpo_df)
    print(result)

    poc = result['poc']
    vah = result['vah']
    val = result['val']

    # --- Create Side-by-Side Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))  # 1 row, 2 columns

    # Plot 1: Disintegrated Profile on the left axis (ax1)
    for index, row in disintegrated_tpo_df.iterrows():
        ax1.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax1.set_title("Disintegrated TPO Chart")
    ax1.set_xlabel("Time Period Index")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, disintegrated_tpo_df['datetime'].max() + 1)
    ax1.set_ylim(truncated_df['low'].min() - 1, truncated_df['high'].max() + 1)

    ax1.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax1.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax1.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax1.legend()

    # Plot 2: Consolidated Profile on the right axis (ax2)
    for index, row in consolidated_tpo_df.iterrows():
        ax2.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax2.set_title("Consolidated Market Profile")
    ax2.set_xlabel("TPO Count")
    ax2.set_ylabel("Price")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(-1, consolidated_tpo_df['datetime'].max() + 1)
    ax2.set_ylim(truncated_df['low'].min() - 1, truncated_df['high'].max() + 1)

    ax2.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax2.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax2.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax2.legend()

    plt.tight_layout()  # Adjusts plots to prevent overlap
    plt.show()

# both profiles side by side, with POC, VAL, VAH
elif CONFIG["market_profile_type"] == 3:

    truncated_df['tpo'] = ''
    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for i in range(len(truncated_df)):
        truncated_df.loc[truncated_df.index[i], 'tpo'] = tpo_letters[i]

    print('df after assigning TPOs')
    print(truncated_df[['date', 'tpo']])

    # --- Generate Data for Disintegrated Profile (Plot 1) ---
    disintegrated_tpo_df = pd.DataFrame(columns=['datetime', 'price', 'tpo'])
    counter = 0
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']
        tick_size = 0.25
        price_range = int((high - low) / tick_size)
        low = math.ceil(low)
        for number in range(price_range):
            if number % 4 == 0 and low < high:
                new_row = {'datetime': index, 'price': low + (number * tick_size), 'tpo': row['tpo']}
                disintegrated_tpo_df = pd.concat([disintegrated_tpo_df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Generate Data for Consolidated Profile (Plot 2) ---
    consolidated_tpo_df = pd.DataFrame(columns=['datetime', 'price', 'tpo'])
    price_level_occupancy = {}
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']
        tick_size = 1.0
        low = math.ceil(low)
        price_points = np.arange(low, high, tick_size)
        for price in price_points:
            price = round(price / tick_size) * tick_size
            x_pos = price_level_occupancy.get(price, 0)
            new_row = {'datetime': x_pos, 'price': price, 'tpo': row['tpo']}
            consolidated_tpo_df = pd.concat([consolidated_tpo_df, pd.DataFrame([new_row])], ignore_index=True)
            price_level_occupancy[price] = x_pos + 1

    # get key levels before plotting
    result = calculate_market_profile_levels(consolidated_tpo_df)
    print(result)

    poc = result['poc']
    vah = result['vah']
    val = result['val']

    # --- Create Side-by-Side Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))  # 1 row, 2 columns

    # Plot 1: Disintegrated Profile on the left axis (ax1)
    for index, row in disintegrated_tpo_df.iterrows():
        ax1.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax1.set_title("Disintegrated TPO Chart")
    ax1.set_xlabel("Time Period Index")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, disintegrated_tpo_df['datetime'].max() + 1)
    ax1.set_ylim(truncated_df['low'].min() - 1, truncated_df['high'].max() + 1)

    ax1.axhspan(val, vah, color='gray', alpha=0.3, label='Value Area (70%)')
    ax1.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax1.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax1.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax1.legend()

    # Plot 2: Consolidated Profile on the right axis (ax2)
    for index, row in consolidated_tpo_df.iterrows():
        ax2.scatter(x=row['datetime'], y=row['price'], marker='s', s=150, c='blue')
    ax2.set_title("Consolidated Market Profile")
    ax2.set_xlabel("TPO Count")
    ax2.set_ylabel("Price")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(-1, consolidated_tpo_df['datetime'].max() + 1)
    ax2.set_ylim(truncated_df['low'].min() - 1, truncated_df['high'].max() + 1)

    ax2.axhspan(val, vah, color='gray', alpha=0.3, label='Value Area (70%)')
    ax2.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax2.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax2.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax2.legend()

    plt.tight_layout()  # Adjusts plots to prevent overlap
    plt.show()

# coloured TPOs
elif CONFIG["market_profile_type"] == 4:

    truncated_df['tpo'] = ''
    tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for i in range(len(truncated_df)):
        truncated_df.loc[truncated_df.index[i], 'tpo'] = tpo_letters[i]

    print('df after assigning TPOs')
    print(truncated_df[['date', 'tpo']])

    # --- Generate Data for Disintegrated Profile (Plot 1) ---
    disintegrated_tpo_df = pd.DataFrame(columns=['datetime', 'price', 'tpo'])
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']
        tick_size = 0.25
        price_range = int((high - low) / tick_size)
        low = math.ceil(low)
        for number in range(price_range):
            if number % 4 == 0 and low < high:
                new_row = {'datetime': index, 'price': low + (number * tick_size), 'tpo': row['tpo']}
                disintegrated_tpo_df = pd.concat([disintegrated_tpo_df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Generate Data for Consolidated Profile (Plot 2) ---
    consolidated_tpo_df = pd.DataFrame(columns=['datetime', 'price', 'tpo'])
    price_level_occupancy = {}
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']
        tick_size = 1.0
        low = math.ceil(low)
        price_points = np.arange(low, high, tick_size)
        for price in price_points:
            price = round(price / tick_size) * tick_size
            x_pos = price_level_occupancy.get(price, 0)
            new_row = {'datetime': x_pos, 'price': price, 'tpo': row['tpo']}
            consolidated_tpo_df = pd.concat([consolidated_tpo_df, pd.DataFrame([new_row])], ignore_index=True)
            price_level_occupancy[price] = x_pos + 1

    # get key levels before plotting
    results = calculate_market_profile_levels(consolidated_tpo_df)
    poc = results['poc']
    vah = results['vah']
    val = results['val']

    # --- Create Side-by-Side Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 10), sharey=True)

    # Plot 1: Disintegrated Profile on the left axis (ax1)
    for index, row in disintegrated_tpo_df.iterrows():
        ax1.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax1.set_title("Disintegrated TPO Chart")
    ax1.set_xlabel("Time Period Index")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, disintegrated_tpo_df['datetime'].max() + 1)
    ax1.set_ylim(truncated_df['low'].min() - 1, truncated_df['high'].max() + 1)
    ax1.axhspan(val, vah, color='gray', alpha=0.2)
    ax1.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax1.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax1.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax1.legend()

    # Plot 2: Consolidated Profile on the right axis (ax2)
    for index, row in consolidated_tpo_df.iterrows():
        # --- NEW: Determine color based on price position relative to Value Area ---
        if val <= row['price'] <= vah:
            color = 'green'  # Inside Value Area
        else:
            color = 'blue'  # Outside Value Area

        ax2.scatter(x=row['datetime'], y=row['price'], marker='s', s=100, c=color)

    ax2.set_title("Consolidated Market Profile")
    ax2.set_xlabel("TPO Count")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(-1, consolidated_tpo_df['datetime'].max() + 1)
    ax2.axhspan(val, vah, color='gray', alpha=0.2, label='Value Area (70%)')
    ax2.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax2.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax2.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

elif CONFIG["market_profile_type"] == 5:
    plot_market_profile(df, CONFIG["start_date"], calculate_market_profile_levels)

elif CONFIG["market_profile_type"] == 6:
    tpo_df_dicts = create_market_profile_coordinates(truncated_df)
    results = calculate_market_profile_levels(tpo_df_dicts['consolidated_tpo_df'])
    poc = results['poc']
    vah = results['vah']
    val = results['val']
    plot_coordinates(tpo_df_dicts['consolidated_tpo_df'], tpo_df_dicts['disintegrated_tpo_df'], 1)