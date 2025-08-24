# this file is an exploration of market profile, as well as market profile concepts
from statsmodels.graphics.tukeyplot import results

from datahandler import DataHandler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime, timedelta

symbols = ["ES=F"]
symbol_to_test = "ES=F"
CONFIG = {
    "symbols": symbols,  # Pass the symbol as a list
    "start_date": "2025-08-01",  # Using a single day for this example
    "end_date": "2025-08-14",
    "interval": "30m",
    "market_profile_type": 10,
    "iteration_choice": 2
}

def truncate_df(df: pd.DataFrame, start_datetime: str, end_datetime: str) -> pd.DataFrame:
    """
    this is a helper function that spits out a truncated dataframe according to the required session
    """
    new_df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]
    new_df = new_df.reset_index()

    return new_df

def get_data(CONFIG: dict) -> dict:
    """
    this function will obtain intraday time price data from the start to the end date,
    beginning with the start date's regular trading hours (0930) and end at the end dates
    overnight trading hours (0900 next day). the function will also do the necessary data
    cleaning as well as separation into overnight and regular trading hours. the output will
    be a dict, with the date, as well as the session as the key, and a pandas dataframe as the value

    since the datahandler method starts at 0400, we will be starting on the regular trading hours
    on the first day and ending with overnight trading hours on the second last day (according to the
    input)
    """
    data_handler = DataHandler(
        symbols=CONFIG["symbols"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        interval=CONFIG["interval"]
    )
    price_data_dict = data_handler.get_data()
    df = price_data_dict[symbol_to_test]  # this is all time price data obtained

    # find number of days:
    start_date = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d")
    number_of_days = (end_date - start_date).days
    print(f'Number of days between {start_date} and {end_date}: {number_of_days}')

    current_day_count = 0  # start with d = 0
    dict = {}

    for i in range(number_of_days):
        # establish beginning date
        start_date = CONFIG["start_date"]
        start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        #print(f'CHECK start date: {start_date_datetime}, format: {type(start_date_datetime)}')

        # calculate day at current iteration, as well as next day, for overnight hours
        current_day = start_date_datetime + pd.Timedelta(days=current_day_count)
        #print(f'CHECK current date: {current_day}, format: {type(current_day)}')

        next_day = current_day + pd.Timedelta(days=1)  # increment by one
        #print(f'CHECK following date: {next_day}, format: {type(next_day)}')

        # convert back to string as needed
        current_day_str = current_day.strftime("%Y-%m-%d")
        next_day_str = next_day.strftime("%Y-%m-%d")

        rth_start = current_day_str + ' 09:30+00:00'
        rth_end = current_day_str + ' 16:00+00:00'

        on_start = current_day_str + ' 16:00+00:00'
        on_end = next_day_str + ' 09:30+00:00'

        # print(f'CHECK rth date: {rth_start}, format: {type(rth_start)}')
        # print(f'CHECK rth date: {rth_end}, format: {type(rth_end)}')
        # print(f'CHECK on_date: {on_start}, format: {type(on_start)}')
        # print(f'CHECK on_date: {on_end}, format: {type(on_end)}')

        rth_df = truncate_df(df, rth_start, rth_end)
        if not rth_df.empty:
            dict.update({rth_start + ' -RTH': rth_df})

        # Get the DataFrame for Overnight Trading Hours
        on_df = truncate_df(df, on_start, on_end)
        if not on_df.empty:
            dict.update({on_start + ' -OVERNIGHT': on_df})
        current_day_count += 1

    return dict

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

def get_key_values(consolidated_tpo_df: pd.DataFrame,
                   calculate_market_profile_levels):
    # --- 5. Get key levels before plotting ---
    results = calculate_market_profile_levels(consolidated_tpo_df)
    poc = results['poc']
    vah = results['vah']
    val = results['val']

def plot_coordinates(disintegrated_tpo_df: pd.DataFrame, consolidated_tpo_df: pd.DataFrame, VAL: int, VAH: int, POC: int,
                     colourVAL: str = 'green'):
    """
    single session plot for both consolidated and disintegrated profiles.
    requires both coordinate dfs of consolidated and disintegrated profiles, as well as key levels
    will not be used for the final iteration of this code
    """
    # --- 6. Create Side-by-Side Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 10), sharey=True)

    vah = VAH
    val = VAL
    poc = POC

    # Plot 1: Disintegrated Profile
    for index, row in disintegrated_tpo_df.iterrows():
        ax1.text(x=row['datetime'], y=row['price'], s=row['tpo'], ha='center', va='center', fontsize=10)
    ax1.set_title("Disintegrated TPO Chart")
    ax1.set_xlabel("Time Period Index")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, disintegrated_tpo_df['datetime'].max() + 1)
    ax1.set_ylim(disintegrated_tpo_df['price'].min() - 1, disintegrated_tpo_df['price'].max() + 1)

    ax1.axhspan(val, vah, color='gray', alpha=0.2)
    ax1.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: {poc}')
    ax1.axhline(vah, color='green', linestyle=':', linewidth=2, label=f'VAH: {vah}')
    ax1.axhline(val, color='blue', linestyle=':', linewidth=2, label=f'VAL: {val}')
    ax1.legend()

    # Plot 2: Consolidated Profile
    for index, row in consolidated_tpo_df.iterrows():
        color = 'green' if val <= row['price'] <= vah else 'blue'
        ax2.scatter(x=row['datetime'], y=row['price'], marker='s', s=100, c=color) # COLOUR LOGIC NOT APPLIED
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


def plot_coordinates_single(consolidated_tpo_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot 2: Consolidated Profile (renamed to avoid confusion with your code)
    for index, row in consolidated_tpo_df.iterrows():
        # color logic goes here
        ax1.scatter(x=row['datetime'], y=row['price'], marker='s', s=10, c='green')  # Use ax1

    ax1.set_title("Consolidated Market Profile")
    ax1.set_xlabel("TPO Count")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1, consolidated_tpo_df['datetime'].max() + 1)

    plt.tight_layout()
    plt.show()


def plot_multi_session_profile(session_data: dict):
    """
    Plots multiple consolidated market profiles side-by-side on a single chart,
    coloring TPOs based on their session's specific Value Area.

    Args:
        session_data (dict): The dictionary containing coordinate dfs and key levels for each session.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    x_offset = 0  # This will track the horizontal position for each new profile

    # Loop through each session (e.g., '2025-08-08-RTH', '2025-08-08-OVERNIGHT', etc.)
    for session_key, session_info in session_data.items():

        # Extract the data for the current session
        coord_df = session_info['coordinate_df']
        poc = session_info['poc']
        vah = session_info['vah']
        val = session_info['val']

        # --- Plot each TPO point for the current session ---
        for index, row in coord_df.iterrows():
            # Determine color based on THIS session's Value Area
            color = 'green' if val <= row['price'] <= vah else 'blue'

            # Plot the point with the calculated horizontal offset
            ax.scatter(
                x=row['datetime'] + x_offset,
                y=row['price'],
                marker='s',
                s=1,  # Smaller size for better detail
                c=color
            )

        # --- Draw the key levels for the current session ---
        profile_width = coord_df['datetime'].max()
        # Use ax.hlines for cleaner horizontal lines across a specific range
        ax.hlines(poc, xmin=x_offset, xmax=x_offset + profile_width, color='red', linestyle='-',
                  label=f'POC ({session_key.split(" ")[-1]})')
        ax.hlines(vah, xmin=x_offset, xmax=x_offset + profile_width, color='lime', linestyle='-')
        ax.hlines(val, xmin=x_offset, xmax=x_offset + profile_width, color='dodgerblue', linestyle='-')

        # Add a vertical line to separate the sessions
        if x_offset > 0:
            ax.axvline(x=x_offset - 2.5, color='black', linestyle='--', alpha=0.1)

        # Update the offset for the next profile, adding a gap of 5 units
        x_offset += profile_width + 5

    ax.set_title("Multi-Session Consolidated Market Profile")
    ax.set_xlabel("TPO Count (Combined Sessions)")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()


results = get_data(CONFIG)
print(results)
#print(results)

counter_one = 2

# before we do that, we need to create the coordinates for all the sessions
# simply loop through them and do the calculations
# the function that calculates the coordinates returns a dict for that session,

# will also calculate the key values for each session

session_coordinates_dict = {}
for key, items in results.items():
    # this loop will calculate the coordinates for the consolidated profile for each session
    # as well as the key values for that session
    # for each session, we will have a dict: the df containing the coordinate data, as well as the key levels

    session_coordinates_df = create_market_profile_coordinates(items)['consolidated_tpo_df']
    session_key_levels_dict = calculate_market_profile_levels(session_coordinates_df)
    session_info_dict = {'coordinate_df': session_coordinates_df,
                         'poc': session_key_levels_dict['poc'],
                         'vah': session_key_levels_dict['vah'],
                         'val': session_key_levels_dict['val']}

    session_coordinates_dict.update({f'{key}': session_info_dict})
    # dict will look like:
    # '2025-08-11 16:00+00:00 -OVERNIGHT' : {'coordinate_df': df, 'poc':, 'vah':, 'val': }

    # print(session_coordinates_dict)
    # print(f'try printing key: {key}') # we can access the key of the dataframe like this:
    # session_coordinates_dict.update({f'{key}': session_coordinates_df,
    #                                  f'{key} key levels: ': calculate_market_profile_levels(session_coordinates_df)})

print('------------------------------------- printing session coordinates dict ------------------------------------------------------------')
print(session_coordinates_dict)
print('------------------------------------- end of session coordinates dict -------------------------------------------------')

plot_multi_session_profile(session_coordinates_dict)


# building coordinates for all required sessions (into one dataframe)
# additionally, we will also be assigning colours to the letter values:
all_coord_df =[]
prev_x_length = 0
counter = 2 # to check for RTH and overnight sessions

run = False # placeholder

if run:
    for value in session_coordinates_dict.values(): # iterate through the values of the dict

        if counter == 2: # on the first iteration, we set the coord df to be this
            all_coord_df = value['coordinate_df']
            print(f'one iteration of RTH? CHECK: PRINTING COORDINATE DF FOR 1st ITERATION::::::::::::::::::::::::::::::::::::::')

            print(value['coordinate_df'])

            # what do we do in one iteration?
            # append the dataframe into the new one
            # the first iteration is the most simple, since we dont need to do any operations

            all_coord_df = pd.concat([all_coord_df, value['coordinate_df']], ignore_index=True)
            print(f'ITERATION CHECK FIRST ITERATION _______________________________: ', counter)
            print(all_coord_df)
            print(f'number of rows:', all_coord_df.shape[0])

            prev_x_length = value['coordinate_df']['datetime'].max()
            print(f'prev_x_length:', prev_x_length)
            # print(value)
            counter += 1

        elif counter % 2 == 0:
            print('one iteration of RTH?')
            # what do we do in one iteration?
            # append the dataframe into the new one
            # the first iteration is the most simple, since we dont need to do any operations

            adjusted_df = value['coordinate_df']
            adjusted_df['datetime'] = value['coordinate_df']['datetime'] + prev_x_length + 5

            all_coord_df = pd.concat([all_coord_df, adjusted_df],ignore_index=True)
            print(f'ITERATION CHECK: ', counter)
            print(all_coord_df)

            print(f'number of rows:', all_coord_df.shape[0])

            prev_x_length = value['coordinate_df']['datetime'].max()
            print(f'prev_x_length:', prev_x_length)

            counter += 1

        else :
            #print('another iteration OVERNIGHT?')
            #print(value)
            counter += 1


# lets try and plot this!
# old plotting function
# plot_coordinates_single(all_coord_df, vah)


#OK NICE!!!!! lets plot ts

# ------------------------------------------ANYTHING BELOW IS IRRELEVANT------------------------------------------

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
#print(tpo_df)

# 2. Separate into regular trading hours and overnight trading hours :-------------------------
start_time = CONFIG["start_date"] + ' 09:30+00:00'
end_time = CONFIG["start_date"] + ' 16:00+00:00'
# print(start_time, end_time)

truncated_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
truncated_df = truncated_df.reset_index(drop=True)
outside_hours_df = df[(df['date'] < start_time) | (df['date'] > end_time)]
outside_hours_df = outside_hours_df.reset_index(drop=True)

#print("TRUNCATED DATA, trading hours")
#print(truncated_df)
#print("OUTSIDE HOURS")
#print(outside_hours_df)


# overall flow of the code:
# 1. get the data, full day intraday data, interval of 30 mins
# 2. separate the data either into regular trading hours or overnight trading hours
# 3. assign letters to the truncated time price data
# 4. create the point dataframe, either for the consolidated or disintegrated market profile
# 5. plot the points
# 5.5. plot value area high value area low

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

        price_range = int((row['high'] - row['low']) / 0.25)  #maximum number of price points (per tick)
        # print(f'no of points for {symbol_to_test} at {index}: {price_range}')
        # print(f'current count: {counter}')

        iteration_choice = CONFIG['iteration_choice']  #plot every point by default
        for number in range(price_range):
            # iterate through the number of price ticks

            print(number)  # just to check
            # we will be doing 3 different ways to plot: every tick, every 2 ticks, every 4 ticks (one point)
            if iteration_choice == 0:  # every tick:
                # can just start with the price as it is
                print(f'Iterating every tick, so low = {low}')

                tpo_df.loc[counter, 'price'] = low + (number * 0.25)
                tpo_df.loc[counter, 'tpo'] = row['tpo']  # current rows' letter
                tpo_df.loc[counter, 'datetime'] = index  # using INDEX instead of DATETIME !!!
                counter += 1

            elif iteration_choice == 1:  # every 2 ticks
                # will have to start with the first .0 or .5
                low = round(low * 2) / 2  # nearest .5
                print(f'iterating every 2 ticks, so low = {low}')

                if number % 2 == 0 and low < high:
                    tpo_df.loc[counter, 'price'] = low + (number * 0.25)
                    tpo_df.loc[counter, 'tpo'] = row['tpo']  # current rows' letter
                    tpo_df.loc[counter, 'datetime'] = index  # using INDEX instead of DATETIME !!!
                    counter += 1

            elif iteration_choice == 2:  # every point:
                # will have to start with the nearest larger whole number
                low = math.ceil(low)  # nearest whole number
                print(f'iteerating every point, so low = {low}')

                if number % 4 == 0 and low < high:
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

    for index, row in truncated_df.iterrows():  #iterate every index (every period)

        high = row['high']
        low = row['low']
        tick_size = 1.0

        low = math.ceil(low)  # nearest whole number
        print(f'iterating every point, so low = {low}')

        # we will be using price per point for the first example
        price_points = (np.arange(low, high, tick_size))
        print(f'price pointes output: {price_points}')

        for price in price_points:  # iterate through price range for the period
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
