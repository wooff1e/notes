import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)		# prevent truncation
pd.set_option('display.max_columns', None)
np.random.seed(0)

csv_path = 'data/dog_breed_photos.csv'

# CSV: Reading
df = pd.read_csv(csv_path, index_col=0)
# CSV: Writing 
output = pd.DataFrame({'PassengerId': test_data.index, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

### Series (a sequence of data values)

pd.Series([1, 2, 3, 4])

pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

### DataFrame  (a table – basically a collection of Series as columns)

pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

df.set_index("title") 

df.Rooms.describe()
df.head()
df.tail()
df.columns
df.shape
df.columnName			# column as a property
df['columnName']		# column as a dictionary value (can handle reserved characters!)
df['columnName'][0]		# first entry of the column
df['critic'] = 'everyone'	# assign value to all entries in a column
df['index_backwards'] = range(len(df), 0, -1)		# add new column

df.points.mean()
id_best = df.points.idxmax()	# returns index of the max value
df.taster_name.unique()		# list of unique values 
df.taster_name.value_counts()	# list of unique values and how often they occur

# iloc - classic python indexing scheme
df.iloc[0]			    # selecting first row
df.iloc[:, 0]			# selecting first column
df.iloc[:3, 0]			# first 3 values of the first column
df.iloc[[0, 1, 2], 0]	# same result
df.iloc[-5:]			# last 5 rows

'''
loc indexes **inclusively** (0:10 will select entries 0,...,10). 
This is bc loc can index any stdlib type: strings, for example. 
If it indexed exclusively like iloc, then to select values [Apples, ..., Potatoes] 
we would have to write df.loc['Apples', 'Potatoet'] 
instead of df.loc['Apples':'Potatoes'] to include the word Potatoes.
'''

df.loc[0, 'country']		# first value of the 'country' column
df.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
df.loc[(df.country == 'Italy') & (df.points >= 90)]
df.loc[(df.country == 'Italy') | (df.points >= 90)]
df.loc[df.country.isin(['Italy', 'France'])]
df.loc[df.price.notnull()]
df.loc[df.price.isnull()]

### Data Types
df.dtypes			# returns dtype of every column
df.price.dtype		# dtype('float64')
df.index.dtype		# dtype('int64')
df.points.astype('float64')
df.select_dtypes(exclude = 'object').describe()
df.select_dtypes(include = ['object']).describe()
# Pandas also supports eg. categorical data and timeseries data. 

# Renaming
df.rename(columns={'points': 'score'})
df.rename(index={0: 'firstEntry', 1: 'secondEntry'})		# separate indexes
df.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns') # captions

# Dropping values
reduced_df = df.drop("Bad_column", axis=1)

### Built-in mapping
counts = df['Breed'].value_counts().rename_axis('breeds').reset_index(name='counts')
small_counts = counts.loc[counts['counts'] <= 2]
other = df.loc[df['Breed'].isin(small_counts['breeds'])]

# this will set new value to a copy which is discarded!
df.loc[other.index]['Breed'] = 'OTHER'

df.loc[other.index, 'Breed'] = 'OTHER'

df.loc[df['Breed'].isin(small_counts.index.tolist())] = 'OTHER'

points_mean = df.points.mean()
df.points - points_mean				    # single value vs column
df.country + " - " + df.region_1		# two columns of the same length
'''
Basic operators are faster than map() or apply() bc they use speed ups built into pandas. However, 
map() and apply() can do more advanced things, like applying conditional logic.
map() takes a function as a parameter (which transforms a single value) and returns a new Series 
where all the values have been transformed by your function.
'''

breeds = df['Breed'].unique().tolist()
classes = {breeds[i] : i for i in range(len(breeds))}

df['Class'] = df['Breed'].map(classes)

review_points_mean = df.points.mean()
df.points.map(lambda p: p - review_points_mean)

# apply() is used if we want to transform a whole DataFrame by calling a custom method on each row. It returns a new DataFrames. 

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

df.apply(remean_points, axis='columns')	# use axis='index' to transform columns

# use progress_apply() to show progress of slow operations
from tqdm import tqdm
tqdm.pandas()

df[['Width', 'Height']] = df.progress_apply(
    lambda row: get_dimensions(row.Image) , axis = 1).tolist()

# 2-d histogram of image width/height
import seaborn as sns 
with sns.axes_style('white'):
    sns.jointplot(x="Width", y="Height", data=df, kind="hist", color ='blue')


### Grouping
# value_counts() that we used before, is just a shortcut to this groupby() operation:

df.groupby('points').points.count()	# get count for each of the point groups
df.groupby('points').price.min()		# cheapest wine in each point value category
best_rating_per_price = df.groupby('price').points.max()

'''
You can think of each group we generate as being a slice of our DataFrame containing only data with values that match. 
This DataFrame is accessible to us directly using the apply() method, and we can then manipulate the data in any way we see fit. 
For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:
'''

df.groupby('winery').apply(lambda df: df.title.iloc[0])
df.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

# groupby() method agg(), alias for aggregate, lets you run a bunch of different functions on your DataFrame simultaneously. 
df.groupby(['country']).price.agg([len, min, max])

# groupby() will sometimes result in what is called a multi-index, which has multiple levels.
countries_reviewed = df.groupby(['country', 'province']).description.agg([len])

mi = countries_reviewed.index
type(mi)		# pandas.core.indexes.multi.MultiIndex
countries_reviewed.reset_index()	  # convert back to normal

# Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. 
# They also require two levels of labels to retrieve a value. 
# Dealing with multi-index output is a common "gotcha" for users new to pandas. 
# Refer to the MultiIndex/Advanced Selection section of the pandas documentation. 

### Sorting
countries_reviewed.sort_values(by='len')
countries_reviewed.sort_values(by='len', ascending=False)
countries_reviewed.sort_values(by=['country', 'len'], ascending=False)

countries_reviewed.sort_index()

# concat() – concatenates elements along an axis, useful if 2 tables have the same columns.
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])

# join() lets you combine different DataFrame objects which have an index in common.
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK')

# The lsuffix and rsuffix parameters are necessary here because the data has the same column names in both British and Canadian datasets. 
# If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.

# There’s also merge(), but most of what it can do can also be done more simply with join(). 

from PIL import Image
# https://www.kaggle.com/enashed/dataset-basic-eda-cleaning
def get_dimensions(path):
    try:
        image = Image.open(path)
        return image.size
    except Exception as e:
        return (0,0)
    
df[['Width', 'Height']] = df.progress_apply(lambda row: get_dimensions(row.Image) , axis = 1).tolist()
df = df[(df['Width'] >= 100) & (df['Height'] >= 100)].copy()

df['AspectRatio'] = df['Width']/df['Height']
df = df[(df['AspectRatio'] > 0.5) & (df['AspectRatio'] < 1.5)].copy()
_ = df[['AspectRatio']].boxplot()

counts = df['Breed'].value_counts()

small_counts = counts.loc[counts.values <= 2]
small_counts.index

df.loc[df['Breed'].isin(small_counts.index.tolist())] = 'OTHER'
df['Breed'].value_counts()


breeds = df['Breed'].unique().tolist()
classes = {breeds[i] : i for i in range(len(breeds))}

df['Class'] = df['Breed'].map(classes)
df


# Categorical features
# Interactions of categorical features can help linear models and KNN

# Get list of categorical variables
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

# 1.Drop Categorical Variables
drop_df = df.select_dtypes(exclude=['object'])
drop_df = df.select_dtypes(exclude=['object'])

# 2.Label Encoding
# Categorical variables that have a clear ordering in the values, are called ordinal variables. Label encoding assigns each unique value to a different integer: 
"Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3)

# Assigning integers at random is a common approach that is simpler than providing custom labels; however, we can expect an additional boost in performance if we provide better-informed labels.
from sklearn.preprocessing import LabelEncoder
label_df = df.copy()
label_df = df.copy()

label_encoder = LabelEncoder()
object_cols = [col for col in df.columns if df[col].dtype == "object"]
for col in object_cols:
    label_df[col] = label_encoder.fit_transform(df[col])
    label_df[col] = label_encoder.transform(df[col])

# Note: Fitting a label encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data. In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them. 
print("Unique values in 'Condition2' column in train set:", df['Condition2'].unique())
print("\nUnique values in 'Condition2' column in val set:", df['Condition2'].unique())

# There are many approaches to fixing this issue. For instance, you can write a custom label encoder to deal with new categories. The simplest approach, however, is to drop the problematic columns.
good_label_cols = [col for col in object_cols if set(df[col]) == set(df[col])]
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped:', bad_label_cols)

label_df = df.drop(bad_label_cols, axis=1)
label_df = df.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in set(good_label_cols):
    label_df[col] = label_encoder.fit_transform(df[col])
    label_df[col] = label_encoder.transform(df[col]) 


# 3.One-Hot Encoding
# One-hot encoding generally does not perform well if the categorical variable takes on 
# a large number of values (i.e., more than 15 different values).

# Cardinality of a categorical variable is the number of unique entries of for that variable. For large datasets, one-hot encoding can greatly expand the size of the dataset, so we typically will only one-hot encode columns with relatively low cardinality. Then, high cardinality columns can either be dropped from the dataset, or we can use label encoding.

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
sorted(d.items(), key=lambda x: x[1]) # Print cardinality by column, in ascending order

low_cardinality_cols = [col for col in object_cols if df[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols) 

# •To use the encoder, we supply only the categorical columns that we want to be one-hot encoded. 
# •handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
# •setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(df[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = df.index
OH_cols_valid.index = df.index

# Remove categorical columns (will replace with one-hot encoding)
num_df = df.drop(object_cols, axis=1)
num_df = df.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df = pd.concat([num_df, OH_cols_train], axis=1)
OH_df = pd.concat([num_df, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_df, OH_df, y_train, y_valid))	# 166089


# Dealing with Missing values
# To select NaN entries you can use pd.isnull() or pd.notnull().
# For technical reasons these NaN values are always of the float64 dtype.
reviews[pd.isnull(reviews.country)]

# get the number of missing data points per column
missing_values_count = data.isnull().sum()		# table with count for every column
missing_values_count = pd.isnull(data).sum()		# same

# percent of data that is missing
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells) * 100

# One of the most important questions: Is this value missing because it wasn't recorded or because it doesn't exist? 
# Depending on this you choose what to do next. The way you deal with missing values affects other preprocessing steps and training. 
# Eg. make sure that you ignore missing values when calculating mean value.

# Reconstruction of the missing value
# Do this before you do feature generation!


# Drop missing values – dropna()
columns_with_na_dropped = df.dropna() 	# drop rows/columns with missing values
columns_with_na_dropped = df.dropna(axis=0) # rows
columns_with_na_dropped = df.dropna(axis=1) # columns 

# just how much data did we lose?
print("Columns in original dataset: %d \n" % data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

# Imputation – in general, avoid doing this before feature generation

# fillna()
reviews.region_2.fillna(0)
# replace with the value that comes next in the column, then replace the remaining na's with 0
data.fillna(method='bfill', axis=0).fillna(0)


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_df = pd.DataFrame(my_imputer.fit_transform(df))
imputed_df = pd.DataFrame(my_imputer.transform(df))

# Imputation removed column names; put them back
imputed_df.columns = df.columns
imputed_df.columns = df.columns


# Sometimes it might make sense to extend your imputation by adding a new column that shows the location of the imputed entries. 
# In some cases, this will meaningfully improve results.
df_plus = df.copy()
df_plus = df.copy()
for col in cols_with_missing:
    df_plus[col + '_was_missing'] = df_plus[col].isnull()
    df_plus[col + '_was_missing'] = df_plus[col].isnull()
# perform imputation



# Parsing Dates
import datetime
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
landslides['date'].dtype	# dtype('O')	where "O" means "object"
print(landslides['date'].head())
# ________________________________
# 0     3/2/07
# 1    3/22/07
# ...
# Name: date, dtype: object

# convert to datetime
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
landslides['date_parsed'].head()
# ________________________________
# 0   2007-03-02
# 1   2007-03-22
# ...
# Name: date_parsed, dtype: datetime64[ns]

# While we're specifying the date format here, sometimes you'll run into an error when there are multiple date formats in a single column. 
# If that happens, you have have pandas try to infer what the right date format should be.
landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

# There are two reasons not to always have pandas guess the time format (using infer_datetime_format = True). 
# The first is that pandas won't always been able to figure out the correct date format, 
# especially if someone has gotten creative with data entry. 
# The second is that it's much slower than specifying the exact format of the dates.

# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# ________________________________
# 0     2.0
# 1    22.0
# ...
# Name: date_parsed, dtype: float64

# One of the biggest dangers in parsing dates is mixing up the months and days. 
# The to_datetime() function does have very helpful error messages, 
# but it doesn't hurt to double-check that the days of the month we've extracted make sense. 
# To do this, let's plot a histogram of the days of the month. 
# We expect it to have values between 1 and 31 and, since there's no reason to suppose the landslides are more common 
# on some days of the month than others, a relatively even distribution. (With a dip on 31 because not all months have 31 days.)

day_of_month_landslides = day_of_month_landslides.dropna()
seaborn.distplot(day_of_month_landslides, kde=False, bins=31)


# Inconsistent Data Entry
# Inconsistencies in capitalizations and trailing white spaces are very common in text data and you can fix a good 80% 
# of your text data entry inconsistencies by making everything lower case and removing any white spaces at the beginning and end of cells.
countries = professors['Country'].unique()
countries.sort()
countries		# ...' Germany', ... 'Germany', 'germany', ...

professors['Country'] = professors['Country'].str.lower()
professors['Country'] = professors['Country'].str.strip()

# Tackling inconsistencies like 'southkorea' and 'south korea': we're going to use the fuzzywuzzy package to help identify 
# which strings are closest to each other. You won't always be able to rely on fuzzy matching 100%, 
# but it will usually end up saving you at least a little time. Fuzzywuzzy returns a ratio given two strings. 
# The closer the ratio is to 100, the smaller the edit distance between the two strings. 
# Here, we're going to get the ten strings from our list of cities that have the closest distance to "d.i khan".

import fuzzywuzzy
from fuzzywuzzy import process
import chardet

matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
______________________
[('south korea', 100),
 ('southkorea', 48),
 ('saudi arabia', 43),
 ('norway', 35),
 ('austria', 33),
 ('ireland', 33),
 ('pakistan', 32),
 ('portugal', 32),
 ('scotland', 32),
 ('australia', 30)]

def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    strings = df[column].unique()
    matches = fuzzywuzzy.process.extract(string_to_match, strings, limit=10,
                                         scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    rows_with_matches = df[column].isin(close_matches)
    df.loc[rows_with_matches, column] = string_to_match
    print("All done!")

replace_matches_in_column(df=professors, column='Country', string_to_match="south korea")