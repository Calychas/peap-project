import click
import pandas as pd
from tqdm.auto import tqdm
import requests


url = "http://localhost:9003"
NOUN_TAGS = {"subst", "depr", "ger"}
VERB_TAGS = {"fin", "praet", "impt", "imps", "inf"}
ADJ_TAGS = {"adj", "adja", "adjp", "adjc", "pact", "ppas"}


def checkKrnntWorking():
    try:
        requests.get(url)
        click.echo(f"Found krnnt tagger at {url}")
    except:
        raise Exception(
            f"Krnnt tagger is not available at {url}. Please run: 'docker run -p 9003:9003 -it -d djstrong/krnnt:1.0.0'"
        )


def get_pos_to_keep(tagset: str) -> set:
    if tagset == "full":
        return set()
    elif tagset == "noun":
        return NOUN_TAGS
    elif tagset == "verb":
        return VERB_TAGS
    elif tagset == "adj":
        return ADJ_TAGS
    else:
        raise Exception(f"Tagset {tagset} not found")


def krnnt_tag(text: str) -> str:
    response = requests.post(url, data=text.encode("utf-8"))
    response_text = response.text.encode("utf-8")
    return response_text.decode("utf-8")


def lemmatize(text: str, pos_to_keep: set, keep_interp: bool) -> str:
    krnnt_text = krnnt_tag(text)

    sentences = krnnt_text.split("\n\n")
    tweet_lemmatized = ""
    for s_idx, sentence in enumerate(sentences):
        if sentence == "":
            continue
        sentence_lines = sentence.split("\n")
        words_data = list(zip(sentence_lines, sentence_lines[1:]))[::2]

        for orth, lex in words_data:
            orth_word, orth_preceding = orth.split("\t")
            lex_lem, lex_tag, _ = lex.split("\t")[1:]

            if lex_tag == "interp":
                pos = None
            else:
                pos = lex_tag.split(":")[0]

            if (pos is None and keep_interp) or (
                pos is not None and (len(pos_to_keep) == 0 or pos in pos_to_keep)
            ):
                if orth_preceding == "space":
                    tweet_lemmatized += " "
                elif orth_preceding == "newline":
                    if s_idx != 0:
                        tweet_lemmatized += "\n"
                elif orth_preceding == "none":
                    pass  # do nothing on purpose
                else:
                    raise Exception(f"Orth preceding not known {orth_preceding}")

                tweet_lemmatized += lex_lem

    return tweet_lemmatized


@click.command()
@click.option(
    "-i",
    "--input-file",
    "input_file",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    required=True,
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
)
@click.option(
    "-t",
    "--tagset",
    "tagset",
    type=click.Choice(["noun", "verb", "adj", "full"]),
    required=True,
    default="full",
)
@click.option("-p", "--punctuation", "interp", type=bool, required=True, default=True)
@click.option("-l", "--lowercase", "lowercase", type=bool, required=True, default=True)
def lemmatize_tweets(
    input_file: str, output_file: str, tagset: str, interp: bool, lowercase: bool
):
    tqdm.pandas()

    click.echo()
    checkKrnntWorking()

    click.echo(f"Reading tweets from {input_file}")
    df: pd.DataFrame = pd.read_pickle(input_file)

    click.echo(
        f"Tweet lemmatization with tagset {tagset}, interp {interp} and lowercase {lowercase}"
    )
    pos_to_keep = get_pos_to_keep(tagset)
    df["tweet"] = df["tweet"].progress_apply(
        lambda x: lemmatize(x, pos_to_keep, interp).lower()
        if lowercase
        else lemmatize(x, pos_to_keep, interp)
    )

    click.echo(f"Saving tweets to {output_file}")
    df.to_pickle(output_file)


if __name__ == "__main__":
    lemmatize_tweets()  # pylint: disable=no-value-for-parameter
