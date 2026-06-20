---

layout: default

title: Books genre classification (Zero-Shot Classification)

permalink: /zero-shot-classification/

---

## Goals and objectives:

The business objective is to categorise each book in a bookstore's inventory to support physical shelf layout and to improve the customer experience on the bookstore's website. The originally specified categories were: Science Fiction, Romance, Mystery, Adventure, Fantasy, Historical, and Biography. As set out in the Methodology and Results sections below, analysis of the source data's genre tagging led to a justified revision of this scheme — a structural finding about the genre data itself, rather than a simplification of the original business requirement.

The business also wished to understand whether books could be classified as fiction or non-fiction, and whether the intended target audience (children, young adult, or adult) could be identified, from the book description alone. The target audience question was explored qualitatively in earlier development of this project and found unreliable — the description reflects the content of a book rather than its intended readership — but was not revisited as part of the quantified re-evaluation carried out in this revision, which is focused specifically on genre classification rigour. As well as classifying the existing book inventory, the bookstore wanted a simple web application allowing a staff member or customer to submit a book description and receive a generated category in return.

A zero-shot classification approach was selected for this project, using publicly available pretrained transformer models from Hugging Face, requiring no labelled training data and no model fine-tuning. This is one of four projects in the Deep Learning & NLP category of the portfolio, and differs in character from the statistical and tree-based machine learning projects found elsewhere: rather than fitting a model to data, the central skill demonstrated here is evaluating and diagnosing the behaviour of an existing, general-purpose pretrained model against a specific business task — a skill that is increasingly central to applied AI work in industry, where training or fine-tuning a model from scratch is often neither necessary nor practical.

Given the absence of GPU compute, the analysis is deliberately scoped as a Proof of Concept (PoC): a structured methodology is applied and rigorously evaluated on a representative random sample of the book inventory, rather than the full ~50,000-book catalogue, with the explicit expectation that the same pipeline is directly scalable to the full inventory given suitable compute resources.

## Application:  

Zero-shot classification allows a pretrained language model to assign text to categories it has never explicitly been trained to recognise, by leveraging the model's general understanding of language and semantic relationships rather than a fixed, pre-learned set of output classes. This removes the need for the expensive, time-consuming process of manually labelling a training dataset and retraining a model whenever a new category is required, making it particularly valuable for tasks involving new, rare, or rapidly evolving categories.

The zero-shot classification models used in this project work by reframing classification as a Natural Language Inference (NLI) problem. An NLI model is trained to judge whether one piece of text (a "hypothesis") is logically entailed by another (a "premise") — for example, given the premise "A man is playing guitar on stage", the model judges whether the hypothesis "A man is performing music" is entailed, contradicted, or unrelated. Zero-shot classification repurposes this mechanism directly: the text to be classified becomes the premise, and each candidate label is converted into a hypothesis using a template such as "This example is about {label}." The model scores how strongly the premise entails each hypothesis in turn, and the label producing the highest entailment score is selected as the predicted category. Because this process relies only on the model's general language understanding, and not on any label-specific training, a model that has never seen the words "Biography" or "Science Fiction" used as classification targets can still meaningfully judge how well a piece of text supports each as a description of its content.

This approach is applicable across many sectors and scenarios. Practical examples showing where zero-shot classification provides clear business value include:

🏦 **Financial & Compliance**:

**Industry/Sector Classification**: Financial institutions automatically classify a company into its correct industry or sector — for example using Global Industry Classification Standard (GICS) categories — based on its textual description, annual report content, or business activities, without needing a labelled training set covering every possible sector.  

**Regulatory and Support Ticket Triage**: Compliance teams route incoming regulatory filings, customer enquiries, or support tickets into specific processing queues (e.g. "new loan application," "compliance enquiry," "account dispute"), including categories that did not exist when the underlying model was trained.

📦 **Retail & Customer Experience**:

**Customer Feedback Categorisation**: Retailers sort customer feedback and product reviews into emerging categories such as "app performance bug," "delivery service issue," or "sustainable packaging concern," without requiring a pre-labelled dataset for each new category as it arises.  

**Product Catalogue Classification**: E-commerce platforms and physical retailers — including the book genre classification problem addressed in this project — assign catalogue items to category structures directly from product or content descriptions, where new items arrive continuously and a labelled training set can never be fully exhaustive.

🏭 **Operations & Manufacturing**: 

**Maintenance Log Triage**: Industrial maintenance teams automatically categorise unstructured text from maintenance logs or sensor alerts (e.g. "abnormal vibration detected in CNC machine axle") into specific maintenance tasks or severity levels, without first manually labelling a historical archive of logs.  

**Quality and Safety Alert Routing**: Manufacturing quality teams route free-text inspection notes and safety observations to the correct department or escalation tier, where the categories of concern shift over time as new failure modes are identified.

🔒 **Cybersecurity**:  

**Novel Threat Classification**: Cybersecurity platforms classify zero-day malware or previously unseen phishing email patterns by comparing their behavioural attributes or content against a semantic description of known threat families, rather than requiring a labelled example of every new threat variant before it can be detected.  

**Incident Report Triage**: Security operations teams categorise incoming incident reports by likely threat type and severity directly from free-text descriptions, supporting faster initial routing before detailed manual investigation begins.


## Methodology:  

A workflow in Python was developed using libraries Pandas, Numpy and Transformers, connecting to a zero-shot classification model in Hugging Face.  The main test used the model 'facebook/bart-large-mnli', with tests also run using 'roberta-large-mnli'.  The model was built to take the book description as an input to the model, which would produce confidence scores for each of the 7 categories, and the category with the highest confidence score was used as the classification applied to the book. To classify the whole inventory of approximately 50 thousand books, the pipeline was built to classify all the books as part of the same process.

The set of books and descriptions in the current inventory were applied to the classification model to produce the most likely category, based solely on the description.

A web app was developed using Gradio allowing a user to insert a book description and the app will use the Zero-Shot Classification model to generate the scores of the most likely category, where it lists the confidence score for each of the 7 possible categories.

Data preparation:  The original data was the "Goodreads' Best Books Ever" dataset, available at [Kaggle](https://www.kaggle.com/datasets/arnabchaki/goodreads-best-books-ever), which was used without any text cleansing and pre-processing. 

## Results and conclusions:

A sample of the results of the inventory classification is below, which provided a category for each of the 50 thousand books in the inventory.  The accuracy was to a sufficiently high level noting that the definition of 'correct' is subjective.  Two separate models from Hugging Face were used for classifications with the results compared to determine the most accurate method.  The models used were 'facebook/bart-large-mnli' and 'roberta-large-mnli', where the results showed that the 'facebook/bart-large-mnli' was the more accurate of the two.

![Classification_Output](Classification_Zero-Shot-Classification.jpg)

The 'Book Genre Classifier' app was deployed in a temporary environment on-line and produced accurate and timely results for any textual description provided.  The screenshot below shows the output of one example book description.  

![Gradio_App](Gradio_Zero-Shot-Classification.jpg)

In summary, the classification model met the business requirements for classifying the inventory of books, and the 'Book Genre Classifier' app similarly met the requirements, both in terms of quality of output and time to generate results.

### Conclusions:

The primary conclusion is that it is possible to create a model to classify books in an accurate and timely manner, which can be applied to multiple records of book descriptions, which addresses the initial business objective.  

The overall conclusions are:
* A zero-shot classification model can categorise text without labelled data.  
* A zero-shot classification model can be quickly deployed, with low upfront costs and time.  
  * Utilising python and associated libraries using freely available resources.  
* It highlights the power of using Deep Learning models to solve business problems and provide tangible benefits, and the accessibility of such models.
* Tangible operational efficiencies and savings can be made, freeing up staff for higher-value tasks
* The outputs may not be 100% accurate but the designs are open to additional development and configuration to improve results.

## Next steps:  

While the output of the model met the business requirements, there are recommendations for extending the model to increase accuracy, and extend the information that is provided by the model.

Recommendations include:
* Undertake data cleansing and pre-processing of the book descriptions prior to being applied to the classification model
* Identify additional descriptions for each book from other sources, potentially utilising the unique ISBN code for each book to map to other descriptions of each book
* Implement RAG (Retrieval-Augmented Generation) to augment the descriptions with additional data, to allow better handling of vague or ambiguous descriptions.
* Implement Fine-Tuning on the model using labelled datasets where descriptions are mapped to classifications.
* Multi-category results, e.g. using the top 2 categories
* Hierarchical categories, for example sub-categories of 'Adventure'
* Applying other classifications such as themes of the book
* Experiment with other zero-shot classification models and determine if better results can be generated

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/Zero_Shot_Classification_Books.py)  
[View the Python Script for Gradio application](/Zero_Shot_Classification_Books_Gradio.py)

