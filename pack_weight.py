import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import requests
import os


def google_sheet_to_csv(
        filename: str,
):
    """
    Download the Google Sheet with pack weight to disk.
    :param filename: Filename to write to.
    :return: None
    """

    # Google Sheet URL
    url = 'https://docs.google.com/spreadsheets/d/1grJrvxfU2_ZW7QsBQBouVJazhB-X7JGaG9omMrKeNDE'

    # If old file exists, remove
    if os.path.exists(filename):
        os.remove(filename)

    # Get
    response = requests.get(f'{url}/export?format=csv')
    assert response.status_code == 200, 'Wrong status code'
    with open(filename, 'wb') as f:
        f.write(response.content)


def get_weight_data(
        cache: bool = True,
        filename: str = 'weights.csv',
) -> pd.DataFrame:
    """
    Get the weights.
    :param cache: Don't download
    :param filename: Filename to write/read to.
    :return: pd.DataFrame
    """

    if not cache:
        google_sheet_to_csv(filename=filename)

    # Get data and return
    df = pd.read_csv(filename)

    # Common cols
    mass = 'Total (g)'
    cat = 'Category'
    name = 'Item'

    # Drop rows where no weight
    df = df[~pd.isna(df[mass])]

    # Drop rows where no category
    df = df[~pd.isna(df[cat])]

    # Get total weights by category, then make category sortable my total mass
    mass_sums = df.groupby(cat).sum()
    cat_list = mass_sums.sort_values(by=mass, ascending=False)[mass].keys()
    cat_order = pd.api.types.CategoricalDtype(cat_list)
    df[cat] = df[cat].astype(cat_order)
    df = df.sort_values(cat)

    # Sort by cat, then mass
    df.sort_values(by=[cat, mass], ascending=[False, True], inplace=True, ignore_index=True)

    return df


def make_plot(
        cache: bool = True,
):
    """
    Make plot.
    :param cache: Use cache.
    :return: None
    """

    # Common cols
    mass = 'Total (g)'
    cat = 'Category'
    name = 'Item'

    # Colors
    colors = ["#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b", "#bdcf32", "#87bc45", "#27aeef", "#b33dc6"][::-1]

    # Get data, cats
    df = get_weight_data(cache=cache)
    cats = df[cat].unique()
    n_cats = len(cats)

    # Create a colors dict for cat lookup
    cat_colors = {}
    for i, icat in enumerate(cats[::-1]):
        cat_colors[icat] = colors[i]
    df['colors'] = df[cat].map(cat_colors)

    # Make matplotlib objects
    figure: plt.Figure = plt.figure(
        figsize=(7, 10),
        dpi=300,
    )
    spec = figure.add_gridspec(5, 1)
    tot_ax: plt.Axes = figure.add_subplot(spec[0, 0])
    cat_ax: plt.Axes = figure.add_subplot(spec[1:5, 0])

    # Plot totals
    grouped = df.groupby(cat)[mass].sum() / 1000
    x_pos = 0
    for i, cat in enumerate(cats[::-1]):

        # Get iteration values
        i_val = grouped.loc[cat]
        i_color = cat_colors[cat]
        kg = round(i_val, 2)

        # Plot bar
        tot_ax.barh(0, i_val, color=i_color, left=x_pos)

        # Add text
        tot_ax.text(
            x_pos + i_val / 2,
            0.75,
            f'{cat} ({kg} kg)',
            color=i_color,
            rotation=45,
        )

        # Increase baseline
        x_pos += i_val

    # Add markers for target pack mass
    my_mass_kg = 66
    # my_mass_kg = 20
    mass_percent = [0.15, 0.2]
    labels = ['min', 'max']
    pos = ['right', 'left']

    # For min, max
    for i in range(2):

        # Generate label
        i_percent = int(mass_percent[i] * 100)
        i_mass = my_mass_kg * mass_percent[i]
        i_label = f'{labels[i]} ({i_percent}%, {round(i_mass, 2)} kg)'

        # Plot line
        for i_color, i_width in zip(['#FFFFFF', '#777777'], [4, 1.5]):
            tot_ax.plot(
                [i_mass] * 2,
                [-0.8, 0.5],
                lw=i_width,
                color=i_color,
            )

        # Plot label
        tot_ax.text(
            i_mass,
            -1.5,
            i_label,
            horizontalalignment=pos[i],
            color='#777777',
        )

    # Format totals plot
    total_pack_mass = grouped.sum()
    tot_ax.set_xlabel(f'pack mass ({round(total_pack_mass, 2)} kg)')
    tot_ax.spines['left'].set_visible(False)
    tot_ax.set_yticks([])
    tot_ax.set_xlim(0, 15)
    tot_ax.set_ylim(-2, 1)

    # Plot horizontal bar plot
    cat_ax.barh(
        df.index, df[mass],
        color=df['colors'],
    )

    # Format categories
    figure.subplots_adjust(
        left=0.3,
        top=0.85,
        hspace=1,
        right=0.85,
    )
    cat_ax.set_yticks(df.index, df[name])
    cat_ax.set_xlabel('item mass (g)')
    # For both
    for ax in [tot_ax, cat_ax]:

        for pos in ['right', 'top']:
            ax.spines[pos].set_visible(False)

    # Save
    figure.savefig('pack_weight.png')


if __name__ == '__main__':
    make_plot(cache=False)
