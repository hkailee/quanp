# importing library packages
import os, requests
import bs4 as bs
from datetime import datetime
import pandas as pd

# to get and save the companies listed in current sp500 from Wikipedia
def retrieve_wiki_sp500_tickers(output_dir):
    """Retrieve S&P500 companies information from wikipedia page.
    Args: 
    output_dir: str. The output directory of the csv (Can be absolute 
                or relative path).

    Returns:
    dataframe: A dataframe that has same information exported to the
               csv output.
    """
    
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    
    data = []
    data_link = []

    for row in table.findAll('tr')[1:]:
        
        # core elements
        cols = row.findAll('td')
        cols = [element.text.strip() for element in cols]
        data.append([element for element in cols if element])
        
        # SEC_filings link
        cols = row.findAll('td')[2].find('a').get('href')
        data_link.append(cols)

    # build dataframe
    col_names = ['Ticker', 'Security', 'SEC_filings', 'GICS_sector', 'GICS_sub_industry', 
                 'Headquarter_location', 'Date_first_added', 'CIK', 'Founded']
    df = pd.DataFrame(data, columns=col_names)
    
    # append SEC_filings link
    df['SEC_filings_link'] = data_link
    df.drop(columns=['SEC_filings'], inplace=True) # drop non-useful column
    
    # export as csv to date
    todaydate = datetime.now().strftime('%Y%m%d') # define today date
    df.to_csv(os.path.abspath(output_dir) + '/SP500_wiki_{}.csv'.format(todaydate))
    
    return df
