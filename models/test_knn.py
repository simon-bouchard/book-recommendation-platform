#import os, sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from knn_index import get_similar_books, set_book_meta

# Fake metadata for testing
book_meta = {
    1303: ("Test Book", ["test", "placeholder"]),
    4321: ("Another Book", ["example", "fiction"]),
}
set_book_meta(book_meta)

results = get_similar_books(item_idx=1303, top_k=5, method="faiss")

for r in results:
    print(f"{r['title']} (ID: {r['item_idx']}) — {r['score']}")
    print(f"  Subjects: {r['subjects']}")
