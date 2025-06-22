import json
import re
import time
import pickle
from collections import defaultdict
from math import log


class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(lambda: {
            'title_tf': 0, 'body_tf': 0, 'title_positions': [], 'body_positions': []
        }))
        self.document_lengths = {}
        self.doc_count = 0
        self.idf_values = {}

    def _tokenize(self, text):
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text.lower())
        return re.findall(r'[\w\u0600-\u06FF]+', text)

    def build_index(self, data):
        self.doc_count = len(data)

        for doc_id, content in data.items():
            title_tokens = self._tokenize(content.get('title', ''))
            body_tokens = self._tokenize(content.get('body', ''))

            self.document_lengths[doc_id] = {
                'title': len(title_tokens),
                'body': len(body_tokens)
            }

            for i, token in enumerate(title_tokens):
                doc_entry = self.inverted_index[token][doc_id]
                doc_entry['title_tf'] += 1
                doc_entry['title_positions'].append(i)

            for i, token in enumerate(body_tokens):
                doc_entry = self.inverted_index[token][doc_id]
                doc_entry['body_tf'] += 1
                doc_entry['body_positions'].append(i)

        for token, postings in self.inverted_index.items():
            doc_frequency = len(postings)
            idf = log(self.doc_count / (doc_frequency + 1))
            self.idf_values[token] = idf
            for doc_id in postings:
                postings[doc_id]['title_tf_idf'] = postings[doc_id]['title_tf'] * idf
                postings[doc_id]['body_tf_idf'] = postings[doc_id]['body_tf'] * idf

        final_index = {
            token: {
                doc_id: {
                    'title_tf_idf': data['title_tf_idf'],
                    'body_tf_idf': data['body_tf_idf'],
                    'title_positions': data['title_positions'],
                    'body_positions': data['body_positions']
                } for doc_id, data in postings.items()
            } for token, postings in self.inverted_index.items()
        }

        return final_index, self.document_lengths, self.idf_values

    def save_index(self, index, doc_lengths, idf_values, output_file='index_data.pkl'):
        index_data = {
            'inverted_index': index,
            'document_lengths': doc_lengths,
            'idf_values': idf_values
        }
        with open(output_file, 'wb') as f:
            pickle.dump(index_data, f)


def build_inverted_index():
    start_time = time.time()

    try:
        with open('crawled_pages.json', 'r', encoding='utf-8') as f:
            crawled_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading crawled_pages.json: {e}")
        return

    indexer = Indexer()
    index, doc_lengths, idf_values = indexer.build_index(crawled_data)
    indexer.save_index(index, doc_lengths, idf_values)

    metadata = {
        doc_id: {'url': content['url'], 'title': content['title']}
        for doc_id, content in crawled_data.items()
    }
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    end_time = time.time()

    stats = {
        'total_documents': len(crawled_data),
        'unique_terms': len(index),
        'total_time_seconds': round(end_time - start_time, 2)
    }
    with open('indexing_report.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print("Indexing process completed successfully.")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    build_inverted_index()