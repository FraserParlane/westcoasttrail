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
    return df


def make_plot(
        cache: bool = True,
):
    """
    Make plot.
    :param cache: Use cache.
    :return: None
    """

    # Get data
    df = get_weight_data(cache=cache)

    # Common cols
    mass = 'Total (g)'
    cat = 'Category'

    # Drop rows where no weight
    df = df[~pd.isna(df[mass])]

    # Drop rows where no category
    df = df[~pd.isna(df[cat])]

    # Get total weights by category, and sort
    mass_sums = df.groupby(cat).sum()
    cat_list = mass_sums.sort_values(by=mass, ascending=False)[mass].keys()
    cat_order = pd.api.types.CategoricalDtype(cat_list)
    df[cat] = df[cat].astype(cat_order)
    df = df.sort_values(cat)

    # Sort by cat, mass
    df.sort_values(by=[cat, mass], ascending=[True, False], inplace=True)

    # Get unique categories
    categories = df[cat].unique()


    print('a')
    fig = px.bar(df, x=cat, y=mass)
    fig.show()



if __name__ == '__main__':
    make_plot(cache=False)