import json
import re
import time
import pickle
import math
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)


def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def load_pickle_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading {filepath}: {e}")
        return {}


crawled_pages = load_json_file('crawled_pages.json')
index_data = load_pickle_file('index_data.pkl')

inverted_index = index_data.get('inverted_index', {})
document_lengths = index_data.get('document_lengths', {})
idf_values = index_data.get('idf_values', {})


def find_minimal_span(position_lists):
    if not position_lists or any(not lst for lst in position_lists):
        return float('inf')

    all_positions = sorted([
        (pos, term_idx)
        for term_idx, term_positions in enumerate(position_lists)
        for pos in term_positions
    ])

    num_terms = len(position_lists)
    term_counts = [0] * num_terms
    terms_in_window = 0
    left_pointer = 0
    min_span = float('inf')

    for i, (pos, term_idx) in enumerate(all_positions):
        if term_counts[term_idx] == 0:
            terms_in_window += 1
        term_counts[term_idx] += 1

        while terms_in_window == num_terms:
            left_pos, left_term_idx = all_positions[left_pointer]
            min_span = min(min_span, pos - left_pos)

            term_counts[left_term_idx] -= 1
            if term_counts[left_term_idx] == 0:
                terms_in_window -= 1
            left_pointer += 1

    return min_span


class SearchEngine:
    def __init__(self, title_weight=0.7, body_weight=0.3, span_boost_factor=2.0, proximity_power=3.0, max_span_dist=20):
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.span_boost_factor = span_boost_factor
        self.proximity_power = proximity_power
        self.max_span_dist = max_span_dist

    def _tokenize_query(self, query):
        normalized_query = query.replace('ي', 'ی').replace('ك', 'ک')
        clean_query = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', normalized_query.lower())
        return re.findall(r'[\w\u0600-\u06FF]+', clean_query)

    def _find_phrase_in_field(self, terms, doc_id, field_positions_key):
        first_term_positions = inverted_index[terms[0]][doc_id].get(field_positions_key, [])
        for pos in first_term_positions:
            is_match = all(
                (pos + i) in inverted_index[terms[i]][doc_id].get(field_positions_key, [])
                for i in range(1, len(terms))
            )
            if is_match:
                return True
        return False

    def _rank_phrase(self, terms):
        if not terms: return []
        if len(terms) == 1: return self._rank(terms)

        scores = {}
        first_term = terms[0]
        if first_term not in inverted_index: return []

        candidate_docs = inverted_index[first_term].keys()
        for doc_id in candidate_docs:
            if not all(term in inverted_index and doc_id in inverted_index[term] for term in terms):
                continue

            has_title_match = self._find_phrase_in_field(terms, doc_id, 'title_positions')
            has_body_match = self._find_phrase_in_field(terms, doc_id, 'body_positions')

            if has_title_match or has_body_match:
                base_score = sum(
                    inverted_index[term][doc_id].get('title_tf_idf', 0) * self.title_weight +
                    inverted_index[term][doc_id].get('body_tf_idf', 0) * self.body_weight
                    for term in terms
                )
                scores[doc_id] = base_score * 100

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _rank(self, terms):
        if not terms: return []

        candidate_docs = set()
        for term in terms:
            if term in inverted_index:
                candidate_docs.update(inverted_index[term].keys())

        if not candidate_docs: return []

        scores = {}
        for doc_id in candidate_docs:
            data = {'title_score': 0.0, 'body_score': 0.0, 'title_pos': [], 'body_pos': []}
            all_in_title, all_in_body = True, True

            for term in terms:
                term_data = inverted_index.get(term, {}).get(doc_id)
                if not term_data:
                    all_in_title, all_in_body = False, False
                    continue

                data['title_score'] += term_data.get('title_tf_idf', 0)
                data['body_score'] += term_data.get('body_tf_idf', 0)

                title_positions, body_positions = term_data.get('title_positions', []), term_data.get('body_positions',
                                                                                                      [])
                if title_positions:
                    data['title_pos'].append(title_positions)
                else:
                    all_in_title = False
                if body_positions:
                    data['body_pos'].append(body_positions)
                else:
                    all_in_body = False

            title_len = document_lengths.get(doc_id, {}).get('title', 1) or 1
            body_len = document_lengths.get(doc_id, {}).get('body', 1) or 1

            norm_title = data['title_score'] / math.sqrt(title_len)
            norm_body = data['body_score'] / math.sqrt(body_len)

            combined_score = (norm_title * self.title_weight) + (norm_body * self.body_weight)

            prox_bonus, title_match_bonus = 0.0, 0.0
            if all_in_title and len(data['title_pos']) == len(terms):
                span = find_minimal_span(data['title_pos'])
                if span < self.max_span_dist:
                    prox_bonus += (1.0 / (1.0 + span)) ** self.proximity_power * self.title_weight
                if span == len(terms) - 1:
                    title_match_bonus = 50

            if all_in_body and len(data['body_pos']) == len(terms):
                span = find_minimal_span(data['body_pos'])
                if span < self.max_span_dist:
                    prox_bonus += (1.0 / (1.0 + span)) ** self.proximity_power * self.body_weight

            final_score = combined_score * (1 + self.span_boost_factor * prox_bonus) + title_match_bonus
            if final_score > 0:
                scores[doc_id] = final_score

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def search(self, query):
        clean_query = query.strip()
        if clean_query.startswith('"') and clean_query.endswith('"'):
            phrase = clean_query.strip('"')
            terms = self._tokenize_query(phrase)
            return self._rank_phrase(terms), terms
        else:
            terms = self._tokenize_query(clean_query)
            return self._rank(terms), terms


@app.route('/')
def home():
    return render_template('search.html')


def format_results(docs, terms):
    results = []
    for doc_id, score in docs[:20]:
        doc = crawled_pages.get(doc_id, {})
        snippet_text = doc.get('body', '')
        snippet = (snippet_text[:300] + '...') if len(snippet_text) > 300 else snippet_text

        for term in terms:
            snippet = re.sub(f'({re.escape(term)})', r'<strong>\1</strong>', snippet, flags=re.IGNORECASE)

        results.append({
            'title': doc.get('title', 'بدون عنوان'),
            'url': doc.get('url', '#'),
            'snippet': snippet or 'هیچ متنی موجود نیست',
            'score': round(score, 4)
        })
    return results


@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query', '') if request.method == 'POST' else request.args.get('query', '')
    if not query:
        return render_template('search.html')

    start_time = time.time()
    engine = SearchEngine()
    docs, terms = engine.search(query)
    results = format_results(docs, terms)
    elapsed_time = round(time.time() - start_time, 3)

    return render_template('results.html', query=query, results=results, total=len(docs), time=elapsed_time)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)