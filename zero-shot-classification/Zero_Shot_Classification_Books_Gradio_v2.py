# ============================================================
# Book Genre Classifier — Gradio Web App
# ============================================================
# Live demo of zero-shot book genre classification, using the category scheme developed and evaluated in the main
# analysis pipeline (Zero_Shot_Classification_Books.py).
# See the project write-up for full methodology, evaluation results, and known limitations.
#
# Model: facebook/bart-large-mnli
#   Evaluation in the main pipeline found no statistically meaningful accuracy difference between bart-large-mnli and
#   roberta-large-mnli across multiple validation runs — bart is used here mainly for continuity with the project's
#   original framing.
#   Swap MODEL_NAME below to 'roberta-large-mnli' to use the alternative model.
# ============================================================

import gradio as gr
from transformers import logging as hf_logging
from transformers import pipeline

hf_logging.set_verbosity_error()  # Suppress routine model-loading warnings


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = 'facebook/bart-large-mnli'

# Must match the category scheme used in the main analysis pipeline.
# 'Science Fiction and Fantasy' is a merged category — Science Fiction,
# Fantasy, and Adventure showed substantial genre-tag overlap in the
# source data (e.g. 299 of 320 'Science Fiction'-tagged books were also
# tagged 'Fantasy'), consistent with how many bookstores shelve these
# genres together rather than separately.
TARGET_LABELS = [
    'Science Fiction and Fantasy', 'Romance',
    'Mystery', 'Historical', 'Biography'
]

# Below this top-1 confidence score, OR below this margin over the
# second-place category, the result is flagged for human review rather
# than presented as a confident classification. Thresholds are informed
# by the confidence analysis in the main pipeline (mean top-1 confidence
# ~0.40; roughly 40% of classifications fell below a 0.10 margin).
CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD     = 0.10

# Example descriptions for the Gradio UI's clickable example inputs
EXAMPLE_DESCRIPTIONS = [
    ["Lawyer Simon Latch is struggling with debt, gambling issues and an "
     "impending divorce. But when Eleanor Barnett, an 85-year-old widow, "
     "visits his office to secure a new will, it seems his luck has "
     "finally changed: she claims she's sitting on a $20 million fortune "
     "and no one else knows about it.\n\nOnce he's hooked the richest "
     "client of his career, Simon works quietly to keep her wealth under "
     "the radar. But it's a terrible mistake. Hidden secrets have a way "
     "of being found out, and when Eleanor is hospitalised after a car "
     "accident, Simon realises that nothing is as it seems."],

    ["An excavation at the lost gardens of Earlsacre Hall is called to a "
     "halt when a skeleton is discovered under a 300 year old stone "
     "plinth, a corpse that seems to have been buried alive. But DS "
     "Wesley Peterson has little time to indulge in his hobby of "
     "archaeology. He has a more recent murder case to solve. A man has "
     "been found stabbed to death in a caravan at a popular holiday park "
     "and the only clue to his identity is a newspaper cutting about the "
     "restoration of Earlsacre. Does local solicitor Brian Willerby have "
     "the answer? He seems eager to talk to Wesley but before he can "
     "reveal his secret he is found dead during a 'friendly' game of "
     "village cricket, apparently struck by a cricket ball several times "
     "with some force. If Wesley is looking for a demon bowler this "
     "appears to let out most of the village side. But what is it about "
     "Earlsacre Hall that leads people to murder?"],

    ["A killer is on the loose. The bodies are piling up. And Judith is "
     "hiding a deadly secret \u2026\n\nSomeone from Judith's past has "
     "turned up in Marlow and is stirring up trouble. With all the "
     "murders that the Marlow Murder Club have had to solve and her "
     "work setting crosswords, Judith's been too busy to give her old "
     "life much thought. But now it's knocking on her door and won't go "
     "away.\n\nOn top of that, Marlow's celebrities are getting "
     "murdered! When a footballer and a thriller writer are found dead, "
     "Judith, Suzie, and Becks must untangle a web of scandal to find "
     "the killer. But with Judith keeping secrets, the Marlow Murder "
     "Club find themselves drifting apart.\n\nThe pressure is on in "
     "more ways than one \u2026\n\nCan they find the killer and help "
     "Judith in time, or could this be the end of the Marlow Murder "
     "Club?"],
]


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print(f'Loading classifier: {MODEL_NAME} ...')
classifier = pipeline('zero-shot-classification', model=MODEL_NAME, device=-1)
print('Classifier ready.')


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_genre(description):
    """
    Classify a book description into one of the 5 target categories.

    Returns:
      scores  — {category: confidence_score} dict, rendered by gr.Label
      flag    — human-readable note on whether to flag for human review
    """
    if not description or not description.strip():
        return {}, 'Enter a book description above to classify it.'

    result = classifier(description, TARGET_LABELS)
    scores = {label: float(score) for label, score in
              zip(result['labels'], result['scores'])}

    top_score = result['scores'][0]
    margin    = result['scores'][0] - result['scores'][1]

    if top_score < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
        flag = (
            f'\u26a0\ufe0f Low confidence ({top_score:.0%}, margin '
            f'{margin:.0%}) \u2014 recommend flagging for human review.\n'
            f'Evaluation found this model can confuse genres that share '
            f'narrative tone \u2014 e.g. tense or secretive Fantasy/Sci-Fi '
            f'descriptions are sometimes misread as Mystery. A human-in-'
            f'the-loop review step would catch cases like this in a '
            f'production workflow.'
        )
    else:
        flag = f'\u2713 Confident classification ({top_score:.0%}).'

    return scores, flag


# ============================================================
# GRADIO INTERFACE
# ============================================================

iface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Textbox(
        lines=6,
        placeholder='Enter a book description here...',
        label='Book Description'
    ),
    outputs=[
        gr.Label(num_top_classes=5, label='Predicted Category'),
        gr.Textbox(label='Review Flag', interactive=False),
    ],
    examples=EXAMPLE_DESCRIPTIONS,
    title='Book Genre Classifier',
    description=(
        'Zero-shot genre classification using facebook/bart-large-mnli. '
        'Categories were derived from genre-tag overlap analysis of the '
        "Goodreads 'Best Books Ever' dataset. See the full project "
        'write-up for methodology, evaluation results, and known '
        'limitations.'
    ),
)


if __name__ == '__main__':
    iface.launch(share=True)