import click
import plwn
import pandas as pd
from tqdm.auto import tqdm


@click.command()
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
)
def sentiment_from_plwordnet(output_file: str):
    wordnet = plwn.load_default()
    lexical_units = wordnet.lexical_units()
    click.echo("Loaded lexical units from PLWORDNET")

    emotions = []

    for lex_unit in tqdm(lexical_units):
        if lex_unit.is_emotional:
            emotions.append(
                (
                    lex_unit.emotion_example,
                    lex_unit.emotion_markedness.name
                )
            )

            if lex_unit.emotion_example_secondary not in ('', None):
                emotions.append(
                    (
                        lex_unit.emotion_example_secondary,
                        lex_unit.emotion_markedness.name
                    )
                )

    emotions_df = pd.DataFrame.from_records(emotions, columns=['text', 'label'])

    emotions_df['label'].replace({
        'weak_negative': 'negative',
        'strong_negative': 'negative',
        'weak_positive': 'positive',
        'strong_positive': 'positive'
    }, inplace=True)

    click.echo(f"Saving to texts with labels to {output_file}")
    emotions_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    sentiment_from_plwordnet()  # pylint: disable=no-value-for-parameter
