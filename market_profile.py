# this file is an exploration of market profile, as well as market profile concepts

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
    "start_date": "2025-08-07",  # Using a single day for this example
    "end_date": "2025-08-08",
    "interval": "30m",
    "market_profile_type": 1
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

# 2. Truncate dataframe to only include trading hours:-------------------------
start_time = CONFIG["start_date"] + ' 09:30+00:00'
end_time = CONFIG["start_date"] + ' 16:00+00:00'
print(start_time, end_time)

truncated_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
truncated_df = truncated_df.reset_index(drop=True)

print("TRUNCATED DATA, trading hours")
print(truncated_df)

# 3. assign letters to each period
truncated_df['tpo'] = ''
tpo_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
for i in range(len(truncated_df)):
    truncated_df.loc[truncated_df.index[i], 'tpo'] = tpo_letters[i]

print('df after assigning TPOs')
print(truncated_df[['date','tpo']])

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

# 3. creating the coordinates of all points to be plotted: --------------------
# assign: price, datetime (index), letter

profile_type = 0

if CONFIG["market_profile_type"] == 0:
    counter = 0
    for index, row in truncated_df.iterrows():
        high = row['high']
        low = row['low']

        price_range = int((row['high'] - row['low'])/0.25) #maximum number of price points (per tick)
        # print(f'no of points for {symbol_to_test} at {index}: {price_range}')
        # print(f'current count: {counter}')

        iteration_choice = 2 #plot every point by default
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

    print(tpo_df.head)
    print(tpo_df.tail)

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

# 3.5 creating coordinates of points to plot (market profile):

elif CONFIG["market_profile_type"] == 1:
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

    print(tpo_df.head)
    print(tpo_df.tail)

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
