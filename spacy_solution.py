from spacy.matcher import PhraseMatcher
from scipy import spatial

import spacy


# method for reading a pdf file
def readPdfFile():
    return open("text.txt").read()


# customer sentence segmenter for creating spacy document object
def setCustomBoundaries(doc):
    # traversing through tokens in document object
    for token in doc[:-1]:
        if token.text == ";":
            doc[token.i + 1].is_sent_start = True
        if token.text == ".":
            doc[token.i + 1].is_sent_start = False
    return doc


# create spacy document object from pdf text
def getSpacyDocument(pdf_text, nlp):
    main_doc = nlp(pdf_text)  # create spacy document object

    return main_doc


# method for searching keyword from the text
def search_for_keyword(keyword, doc_obj, nlp):
    phrase_matcher = PhraseMatcher(nlp.vocab)
    phrase_list = [nlp(keyword)]
    phrase_matcher.add("Text Extractor", None, *phrase_list)

    matched_items = phrase_matcher(doc_obj)

    matched_text = []
    for match_id, start, end in matched_items:
        text = nlp.vocab.strings[match_id]
        span = doc_obj[start:end]
        matched_text.append(span.sent.text)
    return matched_text


# convert keywords to vector
def createKeywordsVectors(keyword, nlp):
    doc = nlp(keyword)  # convert to document object

    return doc.vector


# method to find cosine similarity
def cosineSimilarity(vect1, vect2):
    # return cosine distance
    return 1 - spatial.distance.cosine(vect1, vect2)


# method to find similar words
def getSimilarWords(keyword, nlp):
    similarity_list = []

    keyword_vector = createKeywordsVectors(keyword, nlp)

    for tokens in nlp.vocab:
        if tokens.has_vector:
            if tokens.is_lower:
                if tokens.is_alpha:
                    similarity_list.append(
                        (tokens, cosineSimilarity(keyword_vector, tokens.vector))
                    )

    similarity_list = sorted(similarity_list, key=lambda item: -item[1])
    similarity_list = similarity_list[:30]

    top_similar_words = [item[0].text for item in similarity_list]

    top_similar_words = top_similar_words[:3]
    top_similar_words.append(keyword)

    for token in nlp(keyword):
        top_similar_words.insert(0, token.lemma_)

    for words in top_similar_words:
        if words.endswith("s"):
            top_similar_words.append(words[0 : len(words) - 1])

    top_similar_words = list(set(top_similar_words))

    top_similar_words = [words for words in top_similar_words]

    return ", ".join(top_similar_words)


if __name__ == "__main__":
    # spacy english model (large)
    nlp = spacy.load("en_core_web_lg")
    # nlp.add_pipe(setCustomBoundaries, before="parser")
    keywords = "how"
    main_doc = nlp(readPdfFile())
    # similar_keywords = getSimilarWords(keywords, nlp)
    print(search_for_keyword(keywords, main_doc, nlp))
