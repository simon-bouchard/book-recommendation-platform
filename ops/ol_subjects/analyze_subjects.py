# ops/ol_subjects/analyze_subjects.py
"""
Reusable module for analyzing Open Library subjects from JSONL.

Can be imported and called iteratively to verify cleaning steps.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import re


class SubjectAnalyzer:
    """Analyze subjects from JSONL file."""
    
    def __init__(self, jsonl_path: Path):
        """
        Initialize analyzer with JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file with format:
                       {"item_idx": 123, "subjects": ["subj1", "subj2"]}
        """
        self.jsonl_path = Path(jsonl_path)
        self._data = None
        self._all_subjects = None
        self._subject_counter = None
    
    def load_data(self):
        """Load data from JSONL file."""
        if self._data is not None:
            return  # Already loaded
        
        print(f"Loading {self.jsonl_path}...")
        
        self._data = []
        all_subjects = []
        
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    self._data.append(record)
                    
                    subjects = record.get("subjects", [])
                    if isinstance(subjects, list):
                        all_subjects.extend(subjects)
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} invalid JSON: {e}")
        
        self._all_subjects = all_subjects
        self._subject_counter = Counter(all_subjects)
        
        print(f"Loaded {len(self._data):,} books with {len(all_subjects):,} subject instances")
    
    def basic_stats(self) -> Dict:
        """Get basic statistics about subject coverage."""
        self.load_data()
        
        books_with_subjects = sum(1 for r in self._data if r.get("subjects"))
        books_without_subjects = len(self._data) - books_with_subjects
        
        subject_counts_per_book = [
            len(r.get("subjects", [])) for r in self._data if r.get("subjects")
        ]
        
        stats = {
            "total_books": len(self._data),
            "books_with_subjects": books_with_subjects,
            "books_without_subjects": books_without_subjects,
            "coverage_pct": (books_with_subjects / len(self._data) * 100) if self._data else 0,
            "total_subject_instances": len(self._all_subjects),
            "unique_subjects": len(self._subject_counter),
            "avg_subjects_per_book": (
                sum(subject_counts_per_book) / len(subject_counts_per_book)
                if subject_counts_per_book else 0
            ),
            "min_subjects": min(subject_counts_per_book) if subject_counts_per_book else 0,
            "max_subjects": max(subject_counts_per_book) if subject_counts_per_book else 0,
        }
        
        return stats
    
    def print_basic_stats(self):
        """Print basic statistics."""
        stats = self.basic_stats()
        
        print("=" * 70)
        print("SUBJECT ANALYSIS - BASIC STATISTICS")
        print("=" * 70)
        print(f"\nCoverage:")
        print(f"  Total books: {stats['total_books']:,}")
        print(f"  With subjects: {stats['books_with_subjects']:,} "
              f"({stats['coverage_pct']:.1f}%)")
        print(f"  Without subjects: {stats['books_without_subjects']:,}")
        
        print(f"\nSubject statistics:")
        print(f"  Total subject instances: {stats['total_subject_instances']:,}")
        print(f"  Unique subjects: {stats['unique_subjects']:,}")
        print(f"  Avg subjects per book: {stats['avg_subjects_per_book']:.1f}")
        print(f"  Min subjects: {stats['min_subjects']}")
        print(f"  Max subjects: {stats['max_subjects']}")
    
    def top_subjects(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top N most common subjects."""
        self.load_data()
        return self._subject_counter.most_common(n)
    
    def print_top_subjects(self, n: int = 20):
        """Print top N most common subjects."""
        print(f"\nTop {n} most common subjects:")
        for subject, count in self.top_subjects(n):
            print(f"  {count:>6,}x  {subject[:70]}")
    
    def find_pattern(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """
        Find all subjects matching a pattern.
        
        Args:
            pattern: String pattern to search for
            case_sensitive: Whether to match case-sensitively
        
        Returns:
            List of matching subjects
        """
        self.load_data()
        
        if case_sensitive:
            return [s for s in self._subject_counter.keys() if pattern in s]
        else:
            pattern_lower = pattern.lower()
            return [s for s in self._subject_counter.keys() if pattern_lower in s.lower()]
    
    def analyze_admin_patterns(self) -> Dict[str, List[str]]:
        """
        Find subjects matching administrative/meta patterns.
        
        These are the patterns the enrichment agent is told to ignore:
        - translation to ...
        - works by ...
        - study guides
        - juvenile literature
        - in literature
        - in art
        """
        self.load_data()
        
        admin_patterns = [
            "translation",
            "in literature",
            "in art",
            "study guides",
            "juvenile literature",
            "works by",
            "criticism and interpretation",
            "review",
            "award"
        ]
        
        results = {}
        for pattern in admin_patterns:
            matches = self.find_pattern(pattern, case_sensitive=False)
            if matches:
                results[pattern] = matches
        
        return results
    
    def print_admin_patterns(self):
        """Print analysis of administrative patterns."""
        results = self.analyze_admin_patterns()
        
        if not results:
            print("\nNo administrative patterns found.")
            return
        
        print("\nAdministrative/meta subjects (flagged for potential removal):")
        for pattern, matches in results.items():
            print(f"\n  Pattern: '{pattern}'")
            print(f"    Count: {len(matches)} unique subjects")
            print(f"    Frequency: {sum(self._subject_counter[s] for s in matches):,} instances")
            print(f"    Examples:")
            for subj in matches[:5]:
                count = self._subject_counter[subj]
                print(f"      {count:>6,}x  {subj[:65]}")
    
    def analyze_loc_format(self) -> List[str]:
        """
        Find subjects in Library of Congress format (containing '--').
        
        Example: "United States -- History -- Civil War, 1861-1865"
        """
        self.load_data()
        return [s for s in self._subject_counter.keys() if " -- " in s]
    
    def print_loc_analysis(self):
        """Print analysis of LoC-formatted subjects."""
        loc_subjects = self.analyze_loc_format()
        
        print(f"\nLibrary of Congress formatted subjects (containing '--'):")
        print(f"  Count: {len(loc_subjects):,} unique subjects")
        print(f"  Frequency: {sum(self._subject_counter[s] for s in loc_subjects):,} instances")
        
        if loc_subjects:
            print(f"  Examples:")
            for subj in loc_subjects[:10]:
                count = self._subject_counter[subj]
                print(f"    {count:>6,}x  {subj[:65]}")
    
    def analyze_length_distribution(self) -> Dict:
        """Analyze character and word length distribution."""
        self.load_data()
        
        char_lengths = [len(s) for s in self._subject_counter.keys()]
        word_lengths = [len(s.split()) for s in self._subject_counter.keys()]
        
        return {
            "char_min": min(char_lengths) if char_lengths else 0,
            "char_max": max(char_lengths) if char_lengths else 0,
            "char_avg": sum(char_lengths) / len(char_lengths) if char_lengths else 0,
            "word_min": min(word_lengths) if word_lengths else 0,
            "word_max": max(word_lengths) if word_lengths else 0,
            "word_avg": sum(word_lengths) / len(word_lengths) if word_lengths else 0,
        }
    
    def print_length_analysis(self):
        """Print length distribution analysis."""
        stats = self.analyze_length_distribution()
        
        print(f"\nLength distribution:")
        print(f"  Characters: {stats['char_min']}-{stats['char_max']} "
              f"(avg: {stats['char_avg']:.1f})")
        print(f"  Words: {stats['word_min']}-{stats['word_max']} "
              f"(avg: {stats['word_avg']:.1f})")
        
        # Show examples of very long subjects
        self.load_data()
        long_subjects = [
            (s, self._subject_counter[s]) 
            for s in self._subject_counter.keys() 
            if len(s) > 100
        ]
        long_subjects.sort(key=lambda x: len(x[0]), reverse=True)
        
        if long_subjects:
            print(f"\n  Very long subjects (>100 chars): {len(long_subjects)}")
            print(f"    Examples:")
            for subj, count in long_subjects[:5]:
                print(f"      {count:>6,}x  {subj[:100]}...")
    
    def analyze_rare_subjects(self, max_count: int = 5) -> List[Tuple[str, int]]:
        """
        Find rare subjects (appearing <= max_count times).
        
        Args:
            max_count: Maximum frequency to consider "rare"
        
        Returns:
            List of (subject, count) tuples
        """
        self.load_data()
        return [(s, c) for s, c in self._subject_counter.items() if c <= max_count]
    
    def print_rare_subjects(self, max_count: int = 5, show_examples: int = 20):
        """Print analysis of rare subjects."""
        rare = self.analyze_rare_subjects(max_count)
        
        print(f"\nRare subjects (appearing â‰¤{max_count} times):")
        print(f"  Count: {len(rare):,} unique subjects")
        print(f"  Frequency: {sum(c for _, c in rare):,} instances")
        
        if rare and show_examples:
            print(f"\n  Random examples:")
            import random
            samples = random.sample(rare, min(show_examples, len(rare)))
            for subj, count in sorted(samples, key=lambda x: x[1], reverse=True):
                print(f"    {count}x  {subj[:70]}")
    
    def full_report(self):
        """Print complete analysis report."""
        self.print_basic_stats()
        self.print_top_subjects(20)
        self.print_admin_patterns()
        self.print_loc_analysis()
        self.print_length_analysis()
        self.print_rare_subjects(max_count=5, show_examples=15)
        print("\n" + "=" * 70)


def analyze_subjects(jsonl_path: Path):
    """
    Convenience function for quick analysis.
    
    Args:
        jsonl_path: Path to JSONL file
    """
    analyzer = SubjectAnalyzer(jsonl_path)
    analyzer.full_report()


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[2]
    JSONL_PATH = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v5_combined.jsonl"
    
    if not JSONL_PATH.exists():
        print(f"File not found: {JSONL_PATH}")
        print("Run convert_subjects_to_jsonl.py first!")
    else:
        analyze_subjects(JSONL_PATH)
