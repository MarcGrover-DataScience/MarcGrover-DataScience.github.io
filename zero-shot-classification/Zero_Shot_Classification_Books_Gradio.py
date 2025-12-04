
import gradio as gr
from transformers import pipeline

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define hierarchical labels
# level1_labels = ["Fiction", "Non-Fiction"]
fiction_labels = ["Science Fiction", "Romance", "Mystery", "Adventure", "Fantasy", "Historical", "Biography"]
# nonfiction_labels = ["Biography", "History", "Self-Help", "Science", "Business"]

def predict_genre(description):
    level2_result = classifier(description, fiction_labels, device =-1)

    # Format output
    output = "**Genres (with confidence scores):**\n"
    for label, score in zip(level2_result['labels'], level2_result['scores']):
        output += f"{label}: {score:.2f}\n"
    return output

# Gradio Interface
iface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Textbox(lines=5, placeholder="Enter book description here..."),
    outputs=gr.Textbox(lines = 15, placeholder="text"),
    title="Book Genre Classifier",
    description="Classify book descriptions into Fiction/Non-Fiction and then sub-genres using zero-shot classification."
)

iface.launch(share=True)



# Lawyer Simon Latch is struggling with debt, gambling issues and an impending divorce. But when Eleanor Barnett, an 85-year-old widow, visits his office to secure a new will, it seems his luck has finally changed: she claims she's sitting on a $20 million fortune and no one else knows about it.
#
# Once he's hooked the richest client of his career, Simon works quietly to keep her wealth under the radar. But it's a terrible mistake. Hidden secrets have a way of being found out, and when Eleanor is hospitalised after a car accident, Simon realises that nothing is as it seems.


# An excavation at the lost gardens of Earlsacre Hall is called to a halt when a skeleton is discovered under a 300 year old stone plinth, a corpse that seems to have been buried alive. But DS Wesley Peterson has little time to indulge in his hobby of archaeology. He has a more recent murder case to solve. A man has been found stabbed to death in a caravan at a popular holiday park and the only clue to his identity is a newspaper cutting about the restoration of Earlsacre. Does local solicitor Brian Willerby have the answer? He seems eager to talk to Wesley but before he can reveal his secret he is found dead during a 'friendly' game of village cricket, apparently struck by a cricket ball several times with some force. If Wesley is looking for a demon bowler this appears to let out most of the village side. But what is it about Earlsacre Hall that leads people to murder?


# A killer is on the loose. The bodies are piling up. And Judith is hiding a deadly secret …
#
# Someone from Judith’s past has turned up in Marlow and is stirring up trouble. With all the murders that the Marlow Murder Club have had to solve and her work setting crosswords, Judith’s been too busy to give her old life much thought. But now it’s knocking on her door and won’t go away.
#
# On top of that, Marlow’s celebrities are getting murdered! When a footballer and a thriller writer are found dead, Judith, Suzie, and Becks must untangle a web of scandal to find the killer. But with Judith keeping secrets, the Marlow Murder Club find themselves drifting apart.
#
# The pressure is on in more ways than one …
#
# Can they find the killer and help Judith in time, or could this be the end of the Marlow Murder Club?



