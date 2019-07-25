import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
        Loads the messages and categories datasets and merge into a single dataframe
        Args:
            messages_filepath: Filepath to the messages dataset in csv
            categories_filepath: Filepath to the categories dataset in csv
        Returns:
            Merged Pandas dataframe
    """

    messages = pd.read_csv('./data/disaster_messages.csv')
    categories = pd.read_csv('./data/disaster_categories.csv')

    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
        Cleans the merged dataset for machine learning pipeline
        Args:
            df: Merged dataframe
        Returns:
            df: Cleaned dataframe
    """
    # create a dataframe of the 36 category columns
    idx = categories['id']
    categories = categories['categories'].str.split(";", expand=True)
    categories['id'] = idx
    # select the first row in categories dataframe
    row = categories.iloc[0][0:-1]
    # Extract a list of new column names for categories by using  lambda function to take
    # everything up to the second to last character of each string by slicing
    category_colnames = row.apply(lambda x: x[0:-2])

    # Rearrange and rename `categories` column to columns
    categories.columns = np.append(category_colnames.values, categories.columns[-1])

    # Convert category values to just numbers 0 or 1
    for column in categories.iloc[:, 0:-1]:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, on='id')

    # Get the duplicates' ids
    unique_id, count = np.unique(df['id'], return_counts=True)
    dup_ids = unique_id[np.where(count > 1)]

    # drop duplicates
    for d in dup_ids:
        df.drop(df[df.id == d][1:].index, inplace=True)

    return df

def save_data(df, database_filename):
    """
        Saves cleaned dataset into SQL database
        Args:
            df:  Cleaned dataframe
            database_filename: Name of the database file
        Returns:
            None
    """

    engine = create_engine('sqlite:///{}'.format(database_file_name))
    db_file_name = database_file_name.split("/")[-1]  # extract file name from file path
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()