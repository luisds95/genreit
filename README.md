# Genre It!
Genre It! is a movie genre classifier tool which uses the title and plot of a movie to determine its genre. 

For example, suppose one wants to identify the genre of **"The Lion King"**. One may google that title and get the plot of the movie: "After the murder of his father, a young lion prince flees his kingdom only to learn the true meaning of responsibility and bravery.". 

Now, we have everything we need to make Genre It! work, just call:

`./genreit -title "The Lion King" -description "After the murder of his father, a young lion prince flees his kingdom only to learn the true meaning of responsibility and bravery."`

Output: `('Drama',)`

According to [IMDB](https://www.imdb.com/title/tt6105098/), this movie can be categorized as Animation, Adventure and Drama (and let's face it, the plot is quite dramatic).

Actual genres are automatically inferred from the training data, in this case, the [MovieLens dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv). By default, not all genres are kept but only those that appear in at least 2% of the movies (note that different datasets and parameters may be set). Thus, a total of 19 genres are used, which are: 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Family', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War' and 'Western'.


## Installation and usage
1. Clone this repository `git clone https://github.com/luisds95/genreit.git`
2. Download [MovieLens dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) and uncompress it in a folder called `data`.
3. Install conda environment `conda env create -f movies.yml`
4. Activate conda environment `conda activate movies`
5. Download additional NLTK files:
````
python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
exit()
````
Done!

**Validation of installation**

To validate, run the following lines. If no error occurs and the results are similar, then success!

`./genreit -title "The Lion King" -description "After the murder of his father, a young lion prince flees his kingdom only to learn the true meaning of responsibility and bravery."`

Output: 
```
('Drama',)
```

`./genreit --benchmark"`

Output:
```
Model "LogisticRegressionCV" f1 score: 0.536
A random genre assignation will have resulted in an f1 score of 0.183
This means a difference of 0.353
```

**Options**

* `-title`, `-t`: Specify the title of movie to classify.
* `-description`, `-d`: default=None, help='Plot of movie to classify.
* `-model_file`: Path to model Pickle file.
* `-config`: Path to config file if a model needs to be trained.
* `--not-save`: Don't save the model.
* `--benchmark`: Test model performance or classify a movie.
* `--force-train`: Train a model even if it is already available as a pickle file.

**Configuration file**

The default configuration is encoded in `configs/default.json`. If a different model/configuration is to be used, then the user has to create the corresponding json file and point it to Genre It!:

`./genreit -config "path/to/new/json.json" -title "This is a title" -description "This is the plot of a movie"`

In general, the configuration file has the following structure:
* **estimator**: str. Currently only "logistic" is supported.
* **estimator_params**: dict. Parameters to pass to the estimator.
* **data_file**: str. Location of the data file.
* **sep**: str. Separator used in the datafile.
* **subset**: list. Columns to load from the datafile. Set to `null` if all.
* **text_vars**: list. Columns with input text.
* **max_tfidf**: list or int. Number of maximum TF-IDF features for each text var.
* **genres_var**: str. Name of variable with genres information in data_file.
* **recombine_vertically**: dict. Used if a variable may be used as an alternative source of text. In the MovieLens case, "tagline" is used as an alternative "overview".
* **min_genre_count**: float. Represents the number or proportion of total samples that a genre needs to have as its frequency to be kept in the "target_genres" variable.

## Current performance and how to improve it
Although it is currently semi-fixed to work with the [MovieLens dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) and a L2-reguralized Logistic Regression, little modification is needed to adjust it to further datasets and models. 

Currently, text is only analysed by extracting TF-IDF features which are good for determining the topic of a text, but it doesn't try to understand the general feeling of the text or the context of a single word, thus not exploiting enough the text.

Even so, Genre It! with its default configuration manages to achieve an f1 score of `~53%`. For comparison, with 19 possible genres, a random classifier gets an f1 score of `~18%` on the MovieLens dataset.

One may test the performance of a new configuration (or dataset) by adding the `--benchmark` flag:

`./genreit --benchmark`

Note that, when benchmarking, both `-title` and `-description` are not required and will be ignored if provided. This will train a model following the given configuration and then calculate and print its f1 score along with the f1 score for the corresponding random model.

Better performance may be achieved by either implementing more extensive and/or deeper data preprocessing by adding or simply using Bag-of-Words, word2vec, GloVe or [ELMo](https://arxiv.org/pdf/1802.05365.pdf). Similarly, pretrained models on a more extensive vocabulary (like [BERT](https://arxiv.org/abs/1810.04805)) may also help boost performance.