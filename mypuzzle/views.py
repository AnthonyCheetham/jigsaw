from flask import request,render_template
from mypuzzle import application

import pandas as pd
import numpy as np

@application.route('/')
def home():
    selected_options={}
    for key in ['animals','art','fantasy','food','landscape','licensed','towns_and_cities']:
        selected_options[key]=""
    for key in ['abstract','collage','hand','photo']:
        selected_options[key]=""
    return render_template("index.html",selected_options=selected_options)

@application.route('/index',methods=['GET','POST'])
def index():
    selected_options={}
    for key in ['animals','art','fantasy','food','landscape','licensed','towns_and_cities']:
        selected_options[key]=""
    for key in ['abstract','collage','hand','photo']:
        selected_options[key]=""
    return render_template("index.html",selected_options=selected_options)

@application.route('/submit',methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        print(request.form)
        print(request.data)
        # print(dir(request))
        # Load the dfs
        product_df = pd.read_csv('clean_product_df.csv')

        ######
        # Art style selection
        ######
        art_style = request.form['selectStyle']

        ######
        # Theme selection
        ######
        theme = request.form['selectTheme']

        # Select the right puzzles
        # Handle the case where either or both category can be "any" (i.e. no selection)
        if art_style == 'any_style' and theme == 'any_theme':
            right_puzzles = np.repeat(True,product_df.shape[0])
            trimmed_df = product_df
        elif art_style == 'any_style':
            right_puzzles = (product_df['theme'] == theme)
            trimmed_df = product_df[right_puzzles]            
        elif theme == 'any_theme':
            right_puzzles = (product_df['art_style'] == art_style)
            trimmed_df = product_df[right_puzzles]
        else:
            right_puzzles = (product_df['theme'] == theme) & (product_df['art_style'] == art_style)
            trimmed_df = product_df[right_puzzles]

        n_puzzles_left = right_puzzles.sum()
        print(np.unique(product_df['art_style']),art_style)
        print(np.unique(product_df['theme']),theme)
        print(n_puzzles_left,'puzzles remaining')

        ######
        # Sort-by features
        ######
        # Order by the selected attribute and put in a list to make it easy
        sort_by = request.form['selectFeature']
        # From the model:
        if sort_by == 'image_quality':
            sort_col = 'score_quality'
        elif sort_by == 'difficulty':
            sort_col = 'score_difficulty' 
        elif sort_by == 'fit_of_pieces':
            sort_col = 'score_fit'
        elif sort_by == 'missing_pieces':
            sort_col = 'score_missing_pieces'
        elif sort_by == 'amazon_rating':
            sort_col = 'star_rating'

        vals_to_sort = trimmed_df[sort_col].values
        sort_ix = np.argsort(vals_to_sort)[::-1] # put into descending order

        # Reorder the output list
        output_list = []
        for ix in range(np.min([n_puzzles_left,60])):
            return_ix = sort_ix[ix]
            output_list.append(trimmed_df.iloc[return_ix])

        # What options were selected for the drop down menus?
        selected_options={}
        for key in ['animals','art','fantasy','food','landscape','licensed','towns_and_cities']:
            selected_options[key]=""
        for key in ['abstract','collage','hand','photo']:
            selected_options[key]=""

        selected_options[theme] = True
        selected_options[art_style] = True
        selected_options[sort_by] = True

        return render_template("output.html",output_list = output_list,selected_options=selected_options)

