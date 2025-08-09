# Taken from the .ipynb only the relevant parts. 

df = pd.read_csv('goodreads_library_export.csv')
read_books_df = df[df['Exclusive Shelf'] == 'read']
