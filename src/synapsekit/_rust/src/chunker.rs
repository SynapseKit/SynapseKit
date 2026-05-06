use pyo3::prelude::*;

/// Apply overlap to a list of chunks by prepending the tail of the previous chunk.
fn apply_overlap(chunks: &[String], chunk_overlap: usize) -> Vec<String> {
    if chunk_overlap == 0 || chunks.len() < 2 {
        return chunks.to_vec();
    }
    let mut overlapped = vec![chunks[0].clone()];
    for i in 1..chunks.len() {
        let prev = &chunks[i - 1];
        let tail_start = prev.len().saturating_sub(chunk_overlap);
        let tail = &prev[tail_start..];
        overlapped.push(format!("{tail}{}", chunks[i]));
    }
    overlapped
}

/// Hard split text into fixed-size chunks with overlap.
fn hard_split(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut result = Vec::new();
    let mut i = 0;
    let len = text.len();
    while i < len {
        let end = (i + chunk_size).min(len);
        // Ensure we don't split in the middle of a UTF-8 char
        let end = find_char_boundary(text, end);
        result.push(text[i..end].to_string());
        i += step;
    }
    result
}

/// Find the nearest char boundary at or before `pos`.
fn find_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut p = pos;
    while p > 0 && !s.is_char_boundary(p) {
        p -= 1;
    }
    p
}

/// Merge parts into chunks, recursively splitting oversized parts using
/// the remaining separators (mirrors Python's RecursiveCharacterTextSplitter).
fn merge_recursive(
    parts: &[&str],
    sep: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    remaining_separators: &[String],
) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for part in parts {
        let candidate = if current.is_empty() {
            part.to_string()
        } else {
            format!("{current}{sep}{part}")
        };

        if candidate.len() <= chunk_size {
            current = candidate;
        } else {
            if !current.is_empty() {
                chunks.push(current);
            }
            if part.len() > chunk_size {
                // Recursively split oversized part with remaining separators
                let sub_chunks =
                    recursive_split_inner(part, chunk_size, chunk_overlap, remaining_separators);
                chunks.extend(sub_chunks);
                current = String::new();
            } else {
                current = part.to_string();
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    apply_overlap(&chunks, chunk_overlap)
}

/// Merge parts for character splitter (no recursive sub-splitting).
fn merge_character(
    parts: &[&str],
    sep: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for part in parts {
        let candidate = if current.is_empty() {
            part.to_string()
        } else {
            format!("{current}{sep}{part}")
        };

        if candidate.len() <= chunk_size {
            current = candidate;
        } else {
            if !current.is_empty() {
                chunks.push(current);
            }
            if part.len() > chunk_size {
                // Hard split oversized part
                let sub = hard_split(part, chunk_size, chunk_overlap);
                chunks.extend(sub);
                current = String::new();
            } else {
                current = part.to_string();
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    apply_overlap(&chunks, chunk_overlap)
}

/// Internal recursive split — tries each separator in order.
fn recursive_split_inner(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    separators: &[String],
) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    for (idx, sep) in separators.iter().enumerate() {
        let parts: Vec<&str> = text.split(sep.as_str()).collect();
        if parts.len() > 1 {
            let remaining = &separators[idx + 1..];
            return merge_recursive(&parts, sep, chunk_size, chunk_overlap, remaining);
        }
    }

    hard_split(text, chunk_size, chunk_overlap)
}

#[pyfunction]
pub fn recursive_split(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
) -> Vec<String> {
    recursive_split_inner(text, chunk_size, chunk_overlap, &separators)
}

#[pyfunction]
pub fn character_split(
    text: &str,
    separator: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let parts: Vec<&str> = text.split(separator).collect();
    if parts.len() <= 1 {
        return hard_split(text, chunk_size, chunk_overlap);
    }

    merge_character(&parts, separator, chunk_size, chunk_overlap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(recursive_split_inner("", 100, 0, &[]), Vec::<String>::new());
    }

    #[test]
    fn test_short_text() {
        let result = recursive_split_inner("hello", 100, 0, &[]);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_paragraph_split() {
        let text = "para one\n\npara two\n\npara three";
        let seps = vec!["\n\n".to_string(), "\n".to_string(), " ".to_string()];
        let chunks = recursive_split_inner(text, 15, 0, &seps);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.contains("para one")));
        assert!(chunks.iter().any(|c| c.contains("para three")));
    }

    #[test]
    fn test_recursive_fallback_to_next_separator() {
        // First separator doesn't split, second one does
        let text = "word1 word2 word3 word4 word5";
        let seps = vec!["\n\n".to_string(), " ".to_string()];
        let chunks = recursive_split_inner(text, 12, 0, &seps);
        assert!(chunks.len() >= 2);
        // Each chunk should be <= chunk_size
        for chunk in &chunks {
            assert!(chunk.len() <= 12, "chunk too long: {:?}", chunk);
        }
    }

    #[test]
    fn test_hard_split() {
        let text = "a".repeat(100);
        let chunks = hard_split(&text, 30, 0);
        assert_eq!(chunks.len(), 4); // 30+30+30+10
        assert!(chunks.iter().all(|c| c.len() <= 30));
    }

    #[test]
    fn test_hard_split_with_overlap() {
        let text = "a".repeat(50);
        let chunks = hard_split(&text, 30, 10);
        // step = 20, so chunks at 0..30, 20..50
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_overlap_applied() {
        let text = "aaaa\n\nbbbb\n\ncccc";
        let seps = vec!["\n\n".to_string()];
        let chunks = recursive_split_inner(text, 6, 2, &seps);
        // Second chunk should start with tail of first
        assert!(chunks.len() >= 2);
        if chunks.len() >= 2 {
            assert!(chunks[1].starts_with(&chunks[0][chunks[0].len() - 2..]));
        }
    }

    #[test]
    fn test_character_split_basic() {
        let text = "line1\nline2\nline3\nline4";
        let chunks = character_split(text, "\n", 12, 0);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_character_split_no_separator() {
        let text = "a".repeat(100);
        let chunks = character_split(&text, "\n", 30, 0);
        assert!(chunks.iter().all(|c| c.len() <= 30));
    }

    #[test]
    fn test_recursive_oversized_part_uses_next_separator() {
        // "big paragraph" that needs sub-splitting by sentence
        let text = "sentence one. sentence two. sentence three\n\nshort";
        let seps = vec![
            "\n\n".to_string(),
            ". ".to_string(),
            " ".to_string(),
        ];
        let chunks = recursive_split_inner(text, 20, 0, &seps);
        // The first paragraph is too long for chunk_size=20, should be split by ". "
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert!(chunk.len() <= 20, "chunk too long: {:?} (len={})", chunk, chunk.len());
        }
    }

    #[test]
    fn test_utf8_safety() {
        // Ensure we don't split in the middle of a multi-byte character
        let text = "🎉".repeat(50); // 4 bytes each = 200 bytes
        let chunks = hard_split(&text, 30, 0);
        for chunk in &chunks {
            // Each chunk should be valid UTF-8 (Rust strings guarantee this,
            // but let's verify our boundary logic)
            assert!(chunk.len() <= 30 || chunk.len() == 32); // may round up to char boundary
            // Should not panic on iteration
            for _ in chunk.chars() {}
        }
    }
}
