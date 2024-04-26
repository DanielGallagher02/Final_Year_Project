# Draft Design Documentation

## Introduction
This section should briefly introduce the design considerations of your project, touching on the main objectives and how the design aligns with your research goals. Mention the importance of design in achieving the precision required for crop prediction models.

## System Requirements

### Hardware Requirements
Detail the hardware components selected for the project, including the MacBook Air M1 and DJI Mini SE drone, and justify why they were chosen with respect to their performance and suitability for the project.

Macbook Air M1 - The main laptio is the Macbook Air M1, which was chosen because of its potent Apple M1 chip. This chip provides effective processing power required to handle sophisticated machine learning algorithms, which is critical to the success of my dissertation. The 8-core CPU and 8GB memory give it the speeds and computational capacity needed for data-intensive tasks in computer vision and artifical intelligence. Its performance-energy effiency ratio makes it especially suitable for my project's high processing requirements. 

DJI Mini SE - The Dji Mini SE drone, selected for its high-resolution imaging capabilities and lightweight design, is essential for data acquistion. It is prefect for gathering comprehensive field data in agricultural because of these features. The drone is designed with precision agrculture applications in mind, as evidenced by its take-off weight and camera specifications. Small-scale farming benefits greatly from the Mini SE's cost-to-performance ratio because it offers cutting-edge features at a resonable cost. Its compatibility with Final Cut Pro X and iMovie gurantees a smotth integration for data processing and analysis with the Macbook Air M1.

### Software Requirements
List the software tools, platforms, and libraries you used, such as Python, TensorFlow, and various data processing libraries. Explain how they support the objectives of the project.

Python 
Visual Studio Code
Keras 
Tensorflow


## Functional and Non-functional Requirements
Describe what the system is designed to do (functional requirements) and how it should perform under various conditions (non-functional requirements like performance, usability, reliability, etc.).

Functional - The functional requirements state what the system should do. These requirements must be implemented for the system to work correctly. 

FR-1 Data Accquisition: The System must automatically collect high-quality image data from various crops using the DJI Mini SE drone or suitable sensors or camera equipment. 

FR-2 Image Processing: Implement algorithms capable of pre-processing images to enchance features relevant to crop prediction, such as edge detection, segmentation, and colour transformation.

FR - 3 Model Training: Faciliate the training of transfer learning models using pre-existing datasets, with capabilities to fine-tune models based on specific crop data. 

FR - 4 Yield Prediction: Accurately predict crop yields based on analysed imagery and detected health conditions of the crops.

FR - 5 Reporting: Generate comprehensieve reports on crop health and yield predictions for use in agricultural planning and decision-making. 

Non-functional - The non-functional requirements helps to report how well the system is performing.

NFR - 1: Performance: The system must process images and return predictions within around a few seconds to faciliate real-time agricultural decision-making. This rapid response time is essential to ensure that the system can be effectively used in a dynamic facing enviornments where timely information is critical for decision-making processes.

NFR - 2: Usability: Designed with an intruitive interface that can be used with minimal training , accommodating users with varried techincal backgrounds. 

NFR - 3: Reliability: Ensure high reliability of yield predictions, with mechanisms to handle errors gracefully. 

NFR - 4: Scalability: Capable of scaling to handle larger datasets and more complex models as the system's usage grows. 

NFR - 5: Maintainability: The system should be easy to update and maintain, with clear documentation and a modular design. 

NFR - 6: Security: Making sure to safeguard data intergrity and privacy, especially if cloud storage or processing is involved. 

NFR - 7: Compatibility: Ensure compatibility with a range of devices and platforms that farmers commonly use. 

## Design Choices

### Model Training Resources
Explain the datasets, training techniques, and computational resources utilized for training your machine learning models.

Data Sources 
Dataset composition
Curation and Pre-processing
Diversity and Representation


### Justification of Selected Components
Provide a rationale behind the selection of specific hardware and software components, emphasizing how they contribute to achieving the project's goals.



## Prototype Development

### Design Prototype
Discuss the design of the prototype, including things like the architecture, user interface, and interaction flow.

### Flowchart Diagram
Include a flowchart to visually represent the system's workflow, showing the sequence of operations and decision-making processes.

## Validation and Adjustment Strategy
Outline the process for validating the model's accuracy, including any cross-validation techniques used and how accuracy targets were established and adjusted based on test results.

## Continuous Improvement and Transparency
Discuss the iterative improvement process and how transparency is maintained in reporting the model's accuracy.

## Conclusion
Summarize the design phase, reflecting on the choices made and discussing how the design lays the foundation for the subsequent testing and evaluation phases.