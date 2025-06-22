Setup and Execution
To use this project, the following steps must be performed in sequence.

1. Install Requirements
First, install all the project's dependencies and required libraries using the requirements.txt file. This ensures your environment has all the necessary tools.
```bash
pip install -r requirements.txt
```

2. Run the Web Crawler
Next, run the crawler.py script. This script is designed to gather data from the web and is configured in this project to download and save the content of 4000 pages. The output of this step is the crawled_pages.json file, which is used in the next step.
```bash
python crawler.py
```

3. Build the Index
Once the data is ready, run the Indexer.py script. This script reads the crawled_pages.json file, processes the text (normalization and tokenization), and builds an advanced inverted index. During this process, TF-IDF scores and the exact position of each word are calculated and stored in the index_data.pkl file.
```bash
python indexer.py
```

4. Run the Search Engine
Finally, run the search_engine.py script. This command starts a local web server (using Flask) that loads the previously built index and provides the user interface for the search engine. After running it, you can begin searching by navigating to http://127.0.0.1:5000 in your web browser.
```bash
python search_engine.py
```