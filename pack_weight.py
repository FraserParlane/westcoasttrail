import pandas as pd
import requests
import os


def google_sheet_to_csv():
    """
    Download the Google Sheet with pack weight to disk.
    :return: None
    """

    # Google Sheet URL
    url = 'https://docs.google.com/spreadsheets/d/1grJrvxfU2_ZW7QsBQBouVJazhB-X7JGaG9omMrKeNDE'
    filename = 'weights.csv'

    # If old file exists, remove
    if os.path.exists(filename):
        os.remove(filename)

    # Get
    response = requests.get(f'{url}/export?format=csv')
    assert response.status_code == 200, 'Wrong status code'
    with open(filename, 'wb') as f:
        f.write(response.content)





if __name__ == '__main__':
    google_sheet_to_csv()
