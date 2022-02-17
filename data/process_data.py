### Importing packages
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import sys


### Declaring the error message
error_message = '''
Please, provide the filepaths of the messages and categories datasets as the first and second arguments,
respectively, as well as the path of the database to save the cleansed data as the third argument.
Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
'''


### Declaring functions
def load_data(df1_path, df2_path):
    """
    Loads the data used in the model - messages.csv and categories.csv
    Both files are located in the home directory
    
    INPUT: df1_path - The location of messages.csv
           df2_path - The location of categories.csv
    
    OUTPUT: messages and categories sorted by id and ready to be merged
    
    """
    
    ### Loading the messages dataset
    messages = pd.read_csv(df1_path)
    
    ### Sorting the message dataset
    messages = messages.sort_values(by='id')
    
    ### loading the categories dataset
    categories = pd.read_csv(df2_path)

    ### Sorting the categories dataset by id
    categories = categories.sort_values(by='id')
    
    ### Printing the shapes of the DataFrames
    print(f'The messages dataset has {messages.shape[0]} rows and {messages.shape[0]} columns')
    print(f'The categories dataset has {categories.shape[0]} rows and {categories.shape[0]} columns')
    print("Now, let's cleanse the data\n")
    
    return messages, categories

#############################################################################################################

def cleanse_data(messages, categories):
    
    """
    Merges the dataframes messages and categoeies and cleanses the resulting dataframe 
    
    INPUT: messages   - the messages dataframe, which is one the output of the previous function
           categories - the categories dataframe, which is the other output
    
    OUTPUT: df2 - the merged and cleansed dataframe
    """
    
    ### Merging dataframes
    print('Merging dataframes')
    df_merged = pd.merge(messages, categories, how='inner', on='id')
    print(f'The merged dataframe has {df_merged.shape[0]} rows and {df_merged.shape[1]} columns')
    
    
    ### Creating a dataframe of the 36 individual category columns
    print('Wrangling dataframe categories')
    categories = categories['categories'].str.split(';', expand=True)

    ### Selecting the first row of the categories dataframe
    row = categories.head(1)
    row = row.T
    row = row.rename(columns={0:'Categories'})

    ### Extracting a list of new column names for categories.
    category_colnames = row['Categories'].tolist()

    ### Renaming the columns of `categories`
    categories.columns = category_colnames

    ### Setting each value to be the last character of the string and converting column from string to numeric
    for column in categories:
        categories[column] = categories[column].str.slice(-1, ).astype(int)

    ### Renaming columns
    print('Renaming columns in the dataframe categories')
    categories.columns = [elem[:-2] for elem in category_colnames]
  
    ### Dropping the original categories column from `df`
    df_merged = df_merged.drop('categories', axis=1)

    ### Concatenating the original dataframe with the new `categories` dataframe
    print("Concatenating the original dataframe with the new `categories` dataframe")
    df2 = pd.concat([df_merged, categories], axis=1) 

    ### Checking total number of duplicates
    print('Dropping duplicates')
    df2.duplicated().sum()
    
    ### Dropping duplicates
    df2 = df2.drop_duplicates()

    ### Checking the number of rows after removing duplicates
    df2.shape

    ### Checking if there are still duplicates
    df2.duplicated().sum()

    ### Removing NA's
    print("Removing NA's")
    df2 = df2.dropna(subset=categories.columns)

    ### Checking unique values for each 'numerical' column
    print("Checking unique values for each 'numerical' column\n")
    for column in categories.columns:
        print(df2[column].unique())
    print('\n')

    ### Replacing values for the column 'related'
    df2['related'] = df2['related'].replace(2, 0)

    ### Checking unique values for each 'numerical' column after the replacement
    print("Checking unique values for each 'numerical' column after the replacement")
    for column in categories.columns:
        print(df2[column].unique())
    print('\n')  
    print("Dataframe cleansed! Now, let's save it\n")

    return df2

#############################################################################################################

def save_data_to_db(df, table_name, db_path):
    
    """
    Saves the the cleansed dataframe to a SQL database 
    
    INPUT: df       - the cleansed dataframe, resulting from the merge of messages and categories
           filename - the name of dataframe when saved to a table in the SQL database
    
    OUTPUT: None
    
    """
    
    engine = create_engine('sqlite:///{}'.format(db_path))
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    engine.dispose()
    print(f'{table_name} saved to {db_path}')

#############################################################################################################	
	
def main():
    if len(sys.argv) != 4:
        print(error_message)
    
    else:
        df1_path, df2_path, db_path = sys.argv[1:]
        
        print('Calling functions')
        df_messages, df_categories =  load_data('data/disaster_messages.csv', 'data/disaster_categories.csv')
        df_cleansed = cleanse_data(df_messages, df_categories)
        save_data_to_db(df_cleansed, 'disaster_response_mod', db_path)
        print('End of code')
        
if __name__ == "__main__":
    main()