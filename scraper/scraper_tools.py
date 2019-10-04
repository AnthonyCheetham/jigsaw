import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
import pandas as pd

# Functions for navigating Amazon's website
def go_back(driver):
    driver.execute_script("window.history.go(-1)")
    
def go_to_next_page(driver,text=''):
    """Go to the next page of search results """
    try:
        next_button = driver.find_elements_by_class_name('a-last')
        if 'a-disabled' in next_button[0].get_attribute('class'):
            print('Reached end of list '+text)
            return False
        next_button[0].click()
        return True
    except:
        print('Reached end of list '+text)
        return False
    
def wait(amount=2):
    time.sleep(amount)
    
def get_database(fname = 'review_db.csv'):
    try:
        reviews_df = pd.read_csv(fname,index_col=0)
    except:
        print('File not found. Making new database')
        info = {'Name':['fake_entry'],
            'n_reviews':[0],
            'Reviews':[['some','fake','placeholder reviews']],
            'ReviewRatings':[[0,1,2]],
            'Price':['$15.99'],
            'Brand':['FakeBrand'],
            'img_url':["https://images-na.ssl-images-amazon.com/images/I/61o7KJvI3cL._SX300_QL70_.jpg"],
            'details':[''],
            'star_rating':["4.5 out of 5 stars"]}
        reviews_df = pd.DataFrame(info)
        reviews_df = reviews_df.drop(0)
    return reviews_df
    
########################

def get_info(driver):
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    item_info = {}
    # This block of code will help extract the Brand of the item
    for divs in soup.findAll('div', attrs={'class': 'a-box-group'}):
        try:
            item_info['Brand'] = divs['data-brand']
            break
        except:
            pass

    # This block of code will help extract the Product Title of the item
    for spans in soup.findAll('span', attrs={'id': 'productTitle'}):
        name_of_product = spans.text.strip()
        item_info['Name'] = name_of_product
        break

    # This block of code will help extract the price of the item in dollars
    for divs in soup.findAll('div'):
        try:
            price = str(divs['data-asin-price'])
            item_info['Price'] = '$' + price
            break
        except:
            pass

    # This block of code will help extract the image of the item
    for divs in soup.findAll('div', attrs={'id': 'imgTagWrapperId'}):
        for img_tag in divs.findAll('img', attrs={'id': 'landingImage'}):
            item_info['img_url'] = img_tag['data-old-hires']
            break

    # This block of code will help extract the average star rating of the product
    for i_tags in soup.findAll('i',
                               attrs={'data-hook': 'average-star-rating'}):
        for spans in i_tags.findAll('span', attrs={'class': 'a-icon-alt'}):
            item_info['star_rating'] = spans.text.strip()
            break

    # This block of code will help extract the number of customer reviews of the product
    for spans in soup.findAll('span', attrs={'id': 'acrCustomerReviewText'
                              }):
        if spans.text:
            review_count = spans.text.strip()
            item_info['n_reviews'] = review_count
            break

    # This block of code will help extract top specifications and details of the product
    item_info['details'] = []
    for ul_tags in soup.findAll('ul',
                                attrs={'class': 'a-unordered-list a-vertical a-spacing-none'
                                }):
        for li_tags in ul_tags.findAll('li'):
            for spans in li_tags.findAll('span',
                    attrs={'class': 'a-list-item'}, text=True,
                    recursive=False):
                item_info['details'].append(spans.text.strip())
       
        # This block of code will help extract the product description
    item_info['description'] = []
    for ul_tags in soup.findAll('div',
                                attrs={'id': 'productDescription'
                                }):
        for paragraphs in ul_tags.findAll('p'):
                item_info['description'].append(paragraphs.text.strip())
       
    
    

    return item_info