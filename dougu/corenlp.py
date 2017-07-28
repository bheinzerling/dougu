import requests


dependency_types = (
    "basicDependencies",
    "enhancedDependencies",
    "enhancedPlusPlusDependencies")


def get_query_url(
        corenlp_url="http://localhost:9000",
        output_format="json",
        annotators="tokenize,ssplit,pos,lemma,ner,depparse",
        whitespace_tokenized=False,
        ssplit_eolonly=False):
    """Create a URL for querying the CoreNLP REST server."""

    return (
            '{corenlp_url}/?properties={{"annotators":"{annotators}",'
            '"outputFormat":"{output_format}",'
            '"tokenize.whitespace":"{whitespace_tokenized}",'
            '"ssplit.eolonly":"{ssplit_eolonly}",'
            '"depparse.extradependencies":"MAXIMAL"}}').format(
        corenlp_url=corenlp_url,
        annotators=annotators,
        output_format=output_format,
        whitespace_tokenized=str(whitespace_tokenized).lower(),
        ssplit_eolonly=str(ssplit_eolonly).lower())


def annotate_text(text, query_url):
    r = requests.post(query_url, data=text.encode())
    return r.text


def add_token2dep_indexes(annot):
    for sent in annot["sentences"]:
        token2dep = {
            dep_type: {dep["dependent"]: dep for dep in sent[dep_type]}
            for dep_type in dependency_types}
        sent["token2dep"] = token2dep


if __name__ == "__main__":
    text = "Please parse this sentence, thank you."
    query_url = get_query_url()
    print(annotate_text(text, query_url))
