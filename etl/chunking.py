import re

# Sentence splitting regex
SENTENCE_END = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')


def split_into_sentences(text):
    sentences = SENTENCE_END.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(title, body_text, max_words=256, overlap_words=50):
    """Split `title. body_text` into paragraph/sentence-boundary chunks.

    Builds chunks up to `max_words` words, carrying a `overlap_words`-word
    sliding overlap between consecutive chunks so retrieval doesn't lose
    context at chunk boundaries.
    """
    text = f"{title}. {body_text}" if title else body_text
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sent_words = sentence.split()
        if not sent_words:
            continue

        # If a single sentence is longer than max_words, chunk by words
        if len(sent_words) > max_words:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
            for i in range(0, len(sent_words), max_words - overlap_words):
                chunk_w = sent_words[i : i + max_words]
                chunks.append(" ".join(chunk_w))
            continue

        if current_word_count + len(sent_words) > max_words:
            chunks.append(" ".join(current_chunk_sentences))
            # Create overlap from end of current chunk
            overlap_sentences = []
            overlap_count = 0
            for prev_sent in reversed(current_chunk_sentences):
                prev_sent_words = prev_sent.split()
                if overlap_count + len(prev_sent_words) <= overlap_words:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_count += len(prev_sent_words)
                else:
                    break

            current_chunk_sentences = overlap_sentences + [sentence]
            current_word_count = overlap_count + len(sent_words)
        else:
            current_chunk_sentences.append(sentence)
            current_word_count += len(sent_words)

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
