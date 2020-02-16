import ast
from bs4 import BeautifulSoup
from config import Config
import datetime as dt
import dateutil.relativedelta
import gc
import io
import math
import numpy as np
import os
import pandas_datareader.data as web
import pandas_market_calendars as mcal
import pandas as pd
import requests
import re
from time import sleep
import pandas as pd
import requests
from time import sleep
import unicodedata


NYSE_HOLIDAYS = mcal.get_calendar("NYSE").holidays().holidays


def get_ciks(tickers=None):
    """
    get CIK given a company ticker that are needed to download SEC does
    """
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(wiki_url, header=0, index_col=0)[0]
    if tickers:
        df = df.loc[df.index.isin(tickers)]
    df = df.reset_index()
    res = {ticker: cik for ticker, cik in zip(df["Symbol"], df["CIK"])}
    return res


def get_8k_links(ticker2cik, rundate):
    """
    get links to 8K docs given company CIKs and tickers
    """
    df_list = []
    for ticker, cik in ticker2cik.items():
        try:
            base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            payload = {
                "action": "getcompany",
                "CIK": cik,
                "type": "8-K",
                "output": "xml",
                "dateb": rundate.strftime("%Y%m%d")
            }
            sec_response = requests.get(url=base_url, params=payload)
            soup = BeautifulSoup(sec_response.text, "lxml")
            url_list = soup.findAll("filinghref")
            html_list = []
            
            # get html version of links
            for link in url_list:
                link = link.string
                if link.split(".")[len(link.split(".")) - 1] == "htm":
                    txtlink = link + "l"
                    html_list.append(txtlink)

            doc_list = []
            doc_name_list = []
            
            # get links for txt versions of files
            for k in range(len(html_list)):
                txt_doc = html_list[k].replace("-index.html", ".txt")
                doc_name = txt_doc.split("/")[-1]
                doc_list.append(txt_doc)
                doc_name_list.append(doc_name)
                
            # create dataframe of CIK, doc name, and txt link
            df_link = pd.DataFrame({
                "cik": [cik] * len(html_list),
                "ticker": [ticker] * len(html_list),
                "txt_link": doc_list,
                "doc_name": doc_name_list
            })
            df_list.append(df_link)
        except:
            pass
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    print("Get {} links for {} companies!".format(str(len(df)), str(len(tickers) - 1)))
    return df


def get_one_8k_doc(link):
    """
    download a single doc given the link
    """
    r = requests.get(link)
    filing = BeautifulSoup(r.content, "html5lib", from_encoding="ascii")
    submission_dt = filing.find("acceptance-datetime").string[:14]            
    submission_dt = dt.datetime.strptime(submission_dt, "%Y%m%d%H%M%S")
    for section in filing.findAll("html"):
        try:
            # remove tables
            for table in section("table"):
                table.decompose()
            # convert to unicode
            section = unicodedata.normalize("NFKD",section.text)
            section = section.replace("\t", " ").replace("\n", " ").replace("/s", " ").replace("\'", "'")            
        except AttributeError:
            section = str(section.encode("utf-8"))
    filing = "".join((section))
    return filing, submission_dt


def get_8k_docs(tickers, by_date, save=True):
    """
    wraper function that downloads docs given tickers
    """
    # step 1: get CIK given tickers
    ticker2cik = get_ciks(tickers)

    # step 2: get doc links given CIKs and tickers
    df = get_8k_links(ticker2cik, by_date)

    # step 3: get docs given links
    df["text"], df["release_date"] = "", dt.date(1900, 1, 1)
    success_count, failure_count = 0, 0
    for i in range(len(df)):
        try:
            df.loc[df.index[i], "text"], df.loc[df.index[i], "release_date"] = get_one_8k_doc(df.loc[df.index[i], "txt_link"])
            success_count += 1
        except:
            failure_count += 1
        if i % 10 == 9:
            gc.collect()
            print("Downloading {} docs. {} succeeded, {} failed ...".format(i + 1, success_count, failure_count))
    print("Downloading {} docs. {} succeeded, {} failed ...".format(i + 1, success_count, failure_count))
    
    if save:
        df.to_pickle("doc_data.pkl")
    
    return df


def get_financial_data(tickers, start_date, end_date, save=True):    
    """
    get daily stock change data
    """
    df = []
    for ticker in tickers:        
        try:
            data = web.DataReader(ticker, "yahoo", start=str(start_date), end=str(end_date))
            data["ticker"] = ticker
            df.append(data)
            print("Downloaded daily price data for {}.".format(ticker))
        except:
            print("Failed to download daily price data for {}.".format(ticker))            
    df = pd.concat(df)
    df = df.reset_index()
    df.columns = [col.lower() for col in df.columns]
    df["date"] = df["date"].apply(lambda x: x.date())
    if save:
        df.to_pickle("financial_data.pkl")
    return df


def weekday_check(date, incremental):
    """
    move a date to next "good" date if it's weekend/holiday
    """
    while date.isoweekday() > 5 or date.date() in NYSE_HOLIDAYS:
        date = date + dt.timedelta(days=incremental)
    return date


def calculate_pct_change(start_price, end_price):
    """
    calculate percent change given two prices
    """
    pct_change = (end_price - start_price) / start_price
    pct_change = round(pct_change, 4) * 100
    return pct_change


def calculate_returns(df_docs, df_financial, save=True):
    """
    for each doc/row, compute the corresponding normalized price change
    """
    df_docs["price_pct_change"] = 0
    df_docs["index_pct_change"] = 0
    df_docs["normalized_change"] = 0
    df_docs["start_date"] = dt.date(1900, 1, 1)
    df_docs["end_date"] = dt.date(1900, 1, 1)
    
    for i in range(len(df_docs)):
        try:
            row = df_docs.loc[i]
            ticker, release_date = row["ticker"], row["release_date"]
            if release_date > dt.datetime(1900, 1, 1):
                market_close = release_date.replace(hour=16, minute=0, second=0)
                market_open = release_date.replace(hour=9, minute=30, second=0)

                # if report is released after market hours, take change of start date close and release date open        
                if release_date > market_close:
                    start_date = release_date
                    end_date = release_date + dt.timedelta(days=1)
                    end_date = weekday_check(end_date, 1)
                    start_date, end_date = start_date.date(), end_date.date()                
                    start_price_col, end_price_col = "close", "open"    

                # if report is released before market hours, take change of start date's close and release date's open        
                elif release_date < market_open:
                    start_date = release_date + dt.timedelta(days=-1)
                    start_date = weekday_check(start_date, -1)
                    end_date = release_date
                    start_date, end_date = start_date.date(), end_date.date()
                    start_price_col, end_price_col = "close", "open"     

                # if report is released during market hours, use market close        
                else:
                    start_date = release_date
                    end_date = release_date  
                    start_date, end_date = start_date.date(), end_date.date()                
                    start_price_col, end_price_col = "open", "close"       

                start_price = df_financial.loc[(df_financial["ticker"] == ticker) & (df_financial["date"] == start_date), start_price_col].values[0]
                end_price = df_financial.loc[(df_financial["ticker"] == ticker) & (df_financial["date"] == end_date), end_price_col].values[0]
                index_start_price = df_financial.loc[(df_financial["ticker"] == "^GSPC") & (df_financial["date"] == start_date), start_price_col].values[0]
                index_end_price = df_financial.loc[(df_financial["ticker"] == "^GSPC") & (df_financial["date"] == end_date), end_price_col].values[0]

                price_pct_change = calculate_pct_change(start_price, end_price)
                index_pct_change = calculate_pct_change(index_start_price, index_end_price)
                normalized_change = price_pct_change - index_pct_change
                
                df_docs.loc[i, "price_pct_change"] = price_pct_change
                df_docs.loc[i, "index_pct_change"] = index_pct_change
                df_docs.loc[i, "normalized_change"] = normalized_change
                df_docs.loc[i, "start_date"] = start_date
                df_docs.loc[i, "end_date"] = end_date
                                        
        except Exception as e:
            print(str(e))

    if save:
        df_docs.to_pickle("doc_and_financial_data.pkl")
    
    return df_docs


if __name__ == "__main__":
    tickers = [
        "MSFT", "AAPL", "AMZN", "FB", "BRK-B", "JNJ", "JPM", "GOOG", "GOOGL", "XOM", 
        "V", "PG", "BAC", "DIS" , "PFE", "T", "CSCO", "VZ", "MA", "CVX", 
        "UNH", "HD", "MRK", "INTC", "KO", "BA", "CMCSA", "WFC", "PEP", "NFLX", 
        "C", "MCD", "WMT", "ABT", "ADBE", "ORCL", "PYPL", "MDT", "HON", "IBM", 
        "PM", "TMO", "UNP", "CRM", "COST", "ACN", "AVGO", "AMGN", "TXN", "LIN"
    ]
    tickers.append("^GSPC")
    rundate, start_date, end_date = dt.date(2019, 5, 31), dt.date(2010, 1, 1), dt.date(2019, 5, 31)
    df_docs = get_8k_docs(tickers, rundate)
    df_financial = get_financial_data(tickers, start_date, end_date)
    _ = calculate_returns(df_docs, df_financial)
