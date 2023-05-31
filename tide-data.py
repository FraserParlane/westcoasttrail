"""Ingests and formats tide data"""
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np


def proc_data(
        csv_path: str = 'predictions_08525_Port Renfrew_2023-06-25.csv',
) -> pd.DataFrame:
    """
    Process raw data from
    https://tides.gc.ca/en/stations/08525/2023-06-25?tz=PDT&unit=m
    """

    # Read in data
    df = pd.read_csv(csv_path)
    df.rename(columns={'predictions (m)': 'm'}, inplace=True)

    # Make datetime better format
    old_format = '%Y-%m-%d %H:%M %Z'
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, old_format))

    # Note maxima or minima
    direction = 1
    col = 'm'
    m_col = 'Max'
    df[m_col] = ''
    for i in tqdm(range(len(df) - 1)):
        diff = df.loc[i+1][col] - df.loc[i][col]

        # If now decreasing
        if direction > 0 > diff:
            direction = -1
            df.at[i, m_col] = 'H'

        # If now increasing
        elif direction < 0 < diff:
            direction = 1
            df.at[i, m_col] = 'L'

    # Filter for 15 in
    quarter_hour = df['Date'].dt.minute % 30 == 0
    df = df[quarter_hour | (df['Max'] != '')]
    df.reset_index(inplace=True, drop=True)

    # Better format
    new_format = '%m-%d (%a) %I:%M %p'
    df['Date'] = df['Date'].apply(lambda x: x.strftime(new_format))

    # Iteratively make a dataframe of text and values
    rows = []
    min_m = df['m'].min()
    max_m = df['m'].max()
    range_m = max_m - min_m
    cur_date = ''
    for i, row in df.iterrows():

        # Append day
        i_date = row['Date'][:11]
        if cur_date != i_date:
            cur_date = i_date
            rows.append({'str': cur_date, 'val': None})

        # Append entry
        entry_str = row['Date'][12:]
        entry_str += ' {:.2f}'.format(round(row['m'], 2))
        if row['Max'] != '':
            entry_str += f' {row["Max"]}'
        entry_val = (row['m'] - min_m) / range_m
        rows.append({'str': entry_str, 'val': entry_val})

    # Make df and return
    pdf = pd.DataFrame(rows)
    return pdf


def make_figure():
    """Make a figure of the date times"""

    # Get data
    df = proc_data()

    # Make figure objects
    figure: plt.Figure = plt.figure(
        figsize=(8.5, 11)
    )
    ax: plt.Axes = figure.add_subplot()

    # Define the number of rows and cols to plot
    n_row = 75
    n_col = 5

    # Iterate through rows
    for i in range(len(df)):

        # Get x, y pos
        x_pos = (i // n_row) / n_col
        y_pos = 1 - ((i % n_row) / (n_row + 1)) - (1 / n_row)

        # If outside bounds, stop
        if y_pos > 1:
            print('Not all values printed.')
            break

        # Determine formatting
        val = df.loc[i]['val']
        font_dict = {}
        if np.isnan(val):
            font_dict['fontweight'] = 'bold'

        # Plot text
        ax.text(
            x_pos,
            y_pos,
            df.loc[i]['str'],
            transform=ax.transAxes,
            fontsize=8,
            fontdict=font_dict,
        )

        # Plot point
        if not np.isnan(val):

            xp_min = x_pos + 0.6 * (1 / n_col)
            xp_pos = xp_min + 0.38 * val * (1 / n_col)
            yp_pos = y_pos + 0.3 / n_row
            ax.scatter(
                xp_pos,
                yp_pos,
                c='black',
                transform=ax.transAxes,
                s=5,
            )
            ax.plot(
                [xp_min, xp_pos],
                [yp_pos, yp_pos],
                c='black',
                transform=ax.transAxes,
            )

    # Format
    buff = 0.02
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.95,
    )
    for pos in ['top', 'right', 'left', 'bottom']:
        ax.spines[pos].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    figure.savefig('tides.pdf')


if __name__ == '__main__':
    make_figure()
