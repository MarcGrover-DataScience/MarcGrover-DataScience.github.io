---

layout: default

title: Object Classification (Image Classification)

permalink: /image-classification/

---

#### This project is in development

## Goals and objectives:

The business objective is to automatically label images by applying a classification to each.  A proof-of-concept was built to demonstrate the capability, utilising existing image classification models, and existing images.

The result was an application that could accurately classify images without any fine-tuning or transfer learning of the model, which demonstrated the concept and the potential business application and benefits.

## Application:  

Image classification is one of the foundational applications of computer vision, enabling machines to understand and label the content of a visual scene. Its real-world applications are vast, driving automation and decision-making across almost every major sector.  

* **Manufacturing** - primarily used for quality assurance, reducing waste, and automating visual inspection tasks that are tedious or impossible for human eyes to sustain at high speed. Use cases include; defect detection,  quality control, competent sorting and safety compliance.  
* **Technology** - foundational to user interfaces, search engines, and advanced automated systems.  Examples include; Autonomous Vehicles, Image Search & Tagging, Facial Recognition, and Document Classification    
* **Science** - accelerates research, automates diagnostics, and allows for large-scale environmental monitoring.  Key areas of benefits include; Medical Diagnostics, Pathology & Histology, Agriculture (Crop Health), and Astronomy/Biology Research  
* **Retail** - generally focussed on optimising store operations, improving inventory accuracy, and enhancing the customer experience.  Use cases include; Shelf and Planogram Monitoring, Visual Search (E-commerce), Automated Checkout, and Fashion Analysis  

## Methodology:  

A workflow was built in Python using Pillow, Tasnfromers (Hugging Face) and PyTorch libraries.  The Image Classification application uses the *'google/vit-base-patch16-224'* model via Hugging Face, which is a popular Vision Transformer model pre-trained on 14 million images, with additional fine-tuning on a million images.  The Vision Transformer (ViT) is a transformer encoder model based on the BERT architecture.

The application has a basic workflow.  A local image is referenced in the python code, which is then applied to the model, returning the single highest probability classification.  The image is resized to 224x224 pixels and transformed into a PyTorch tensor, to be input into the ViT model, using Pillow.

The app is built so that it can easily be pointed to a library of images, rather than a single image and bulk classification can be applied.

The process returns the image along with the classification applied by the model.

## Results and conclusions:

The results demonstrated both the technical feasibility of the concept as well as the proving the model is capable of producing the required output classifications, applying the correct general category.  

The solution validated that a state-of-the-art, high-performance architecture possibel using  with our existing available and compatible technologies such as Python / PyTorch / Hugging Face.  

It demonstrated that the time to apply classifications is practical even using a local machine with CPU, where there are multiple methods to improve the speed and performance using servers with GPUs and / or cloud services. Each classification in the PoC returns in 2 to 3 seconds, even using a CPU on a local machine, and where the python script classifies a single image.

The classifications by the model applied were considered to be accurate based on human validation, noting that there is always a level of subjectivity to the classifications.  Image below show three examples, where for example an expert baker might suggest that the image of a loafof bread is not in fact a 'French loaf'.  As such providing objective accuracy metrics of the model is complex, but for the sake of a proof-of-concept the results are considered accurate.  It was noted that some misclassifications did occur with visually similar items, but given that the model wasn't fine-tuned this could be considered an expected result.

Example classifications include:

![Banana_classification](ImClas_Bananas.png)  
![Cup_classification](ImClas_Cup.png)  
![Loaf_classification](ImClas_Loaf.png)  

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.  The following a recommendations for next steps in developing an Image Classification solution to meet the next stage in the business objectives.

* Extend the piepline to apply batch classify a library of images, i.e. enable classification of mutliple images within a workflow, rather than classifications on a single image basis
* Migrate to a server using a GPU or a cloud service for faster results
* Leverage Transfer Learning, meaning only training a final classification layer with a relatively small set of proprietary labeled data. This reduces data collection costs and accelerates time-to-value.
* Fine-tune the model using labelled images specific to the business scenario the application is being used for
* Thorough validation and quality control step to confirm the accuracy of the classification, and enable analysis of mis-classifications to support future model fine-tuning


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/ImageClassification_v4.py)
